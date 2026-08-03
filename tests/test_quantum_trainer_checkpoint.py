from __future__ import annotations

import gc
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch
from ultralytics.cfg import get_cfg
from ultralytics.utils import DEFAULT_CFG
from ultralytics.utils.torch_utils import ModelEMA

from src.qvisionframe.quantum_checkpoint import (
    load_quantum_checkpoint,
    reconstruct_quantum_model,
)
from src.qvisionframe.quantum_detection_model import (
    QuantumGatedDetectionModel,
)
from src.qvisionframe.quantum_detection_trainer import (
    QuantumDetectionTrainer,
)


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIRECTORY = (
    ROOT
    / "outputs/inventory/"
    "quantum_trainer_checkpoint_test"
)


def make_batch(
    device: torch.device,
) -> dict[str, torch.Tensor]:
    return {
        "img": torch.rand(
            2,
            3,
            320,
            320,
            device=device,
        ),
        "cls": torch.tensor(
            [[0.0], [5.0], [2.0], [7.0]],
            dtype=torch.float32,
            device=device,
        ),
        "bboxes": torch.tensor(
            [
                [0.30, 0.35, 0.20, 0.25],
                [0.70, 0.65, 0.15, 0.20],
                [0.40, 0.45, 0.25, 0.30],
                [0.75, 0.30, 0.18, 0.22],
            ],
            dtype=torch.float32,
            device=device,
        ),
        "batch_idx": torch.tensor(
            [0, 0, 1, 1],
            dtype=torch.long,
            device=device,
        ),
    }


def clone_state(
    model: torch.nn.Module,
) -> dict[str, torch.Tensor]:
    return {
        key: tensor.detach().cpu().clone()
        for key, tensor
        in model.state_dict().items()
    }


def compare_state(
    expected: dict[str, torch.Tensor],
    actual: dict[str, torch.Tensor],
) -> dict[str, Any]:
    assert expected.keys() == actual.keys()

    exact = 0
    nonexact: list[str] = []
    maximum = 0.0

    for key in expected:
        first = expected[key]
        second = actual[key]

        if (
            first.dtype == second.dtype
            and torch.equal(first, second)
        ):
            exact += 1
        else:
            nonexact.append(key)

        difference = (
            first.float()
            - second.float()
        )

        if difference.numel():
            maximum = max(
                maximum,
                float(
                    difference.abs().max()
                ),
            )

    return {
        "exact": exact,
        "total": len(expected),
        "nonexact": nonexact,
        "max_difference": maximum,
    }


def main() -> None:
    print(
        "===== QUANTUM TRAINER CHECKPOINT TEST ====="
    )

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is unavailable"
        )

    OUTPUT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    weights_directory = (
        OUTPUT_DIRECTORY / "weights"
    )
    weights_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    device = torch.device("cuda:0")

    # Bypass full BaseTrainer initialization so this test isolates
    # the custom get_model/save_model contract.
    trainer = object.__new__(
        QuantumDetectionTrainer
    )

    trainer.quantum_baseline_id = "B6"
    trainer.quantum_target_layer = 10
    trainer.quantum_channels = 256
    trainer.quantum_n_qubits = 4
    trainer.quantum_n_layers = 2
    trainer.quantum_hidden_dim = 64
    trainer.quantum_alpha_init = 1.0e-3
    trainer.quantum_gamma = 0.5

    trainer.data = {
        "nc": 20,
        "names": {
            index: str(index)
            for index in range(20)
        },
    }

    model = trainer.get_model(
        cfg="yolo11n.yaml",
        weights=None,
        verbose=False,
    ).to(device).train()

    model.args = get_cfg(
        cfg=DEFAULT_CFG
    )
    model.criterion = None

    trainer.model = model
    trainer.ema = ModelEMA(model)

    trainer.optimizer = torch.optim.SGD(
        model.parameters(),
        lr=1.0e-3,
        momentum=0.9,
        weight_decay=5.0e-4,
    )

    trainer.scaler = torch.amp.GradScaler(
        "cuda",
        enabled=False,
    )

    trainer.epoch = 0
    trainer.best_fitness = 0.25
    trainer.fitness = 0.25
    trainer.metrics = {
        "metrics/mAP50(B)": 0.0,
    }

    trainer.args = SimpleNamespace(
        model="yolo11n.yaml",
        data="configs/datasets/voc.yaml",
        epochs=1,
        imgsz=320,
        batch=2,
        device=0,
        pretrained=False,
        seed=42,
        deterministic=True,
        plots=False,
        compile=False,
    )

    trainer.wdir = weights_directory
    trainer.last = (
        weights_directory / "last.pt"
    )
    trainer.best = (
        weights_directory / "best.pt"
    )
    trainer.save_period = 1

    trainer.read_results_csv = lambda: {
        "epoch": [0],
        "train/box_loss": [1.0],
    }

    batch = make_batch(device)

    trainer.optimizer.zero_grad(
        set_to_none=True
    )

    loss_components, loss_items = model(
        batch
    )
    loss = loss_components.sum()
    loss.backward()
    trainer.optimizer.step()

    trainer.ema.update(model)

    # Materialize both raw and EMA simulators.
    model.eval()

    with torch.no_grad():
        model(batch["img"])
        trainer.ema.ema(batch["img"])

    model.train()

    raw_state_before = clone_state(model)
    ema_state_before = clone_state(
        trainer.ema.ema
    )

    save_result = trainer.save_model()

    print("save_model result        :", save_result)
    print("last.pt exists           :", trainer.last.exists())
    print("best.pt exists           :", trainer.best.exists())

    assert save_result is True
    assert trainer.last.exists()
    assert trainer.best.exists()

    payload = load_quantum_checkpoint(
        trainer.last,
        map_location="cpu",
    )

    print("Checkpoint format        :", payload["qvf_format"])
    print("Checkpoint epoch         :", payload["epoch"])
    print("Checkpoint baseline      :", payload["model_metadata"]["gate"]["baseline_id"])
    print("Optimizer present        :", payload["optimizer_state_dict"] is not None)
    print("Optimizer state entries  :", len(payload["optimizer_state_dict"]["state"]))

    restored_raw = reconstruct_quantum_model(
        payload,
        state_key="model_state_dict",
        device="cpu",
        strict=True,
        verbose=False,
    )

    restored_ema = reconstruct_quantum_model(
        payload,
        state_key="ema_state_dict",
        device="cpu",
        strict=True,
        verbose=False,
    )

    raw_comparison = compare_state(
        raw_state_before,
        clone_state(restored_raw),
    )

    ema_comparison = compare_state(
        ema_state_before,
        clone_state(restored_ema),
    )

    print(
        "Raw exact state tensors  :",
        f"{raw_comparison['exact']}/"
        f"{raw_comparison['total']}",
    )
    print(
        "EMA exact state tensors  :",
        f"{ema_comparison['exact']}/"
        f"{ema_comparison['total']}",
    )
    print(
        "Raw max state difference :",
        raw_comparison["max_difference"],
    )
    print(
        "EMA max state difference :",
        ema_comparison["max_difference"],
    )
    print(
        "Raw quantum dtype        :",
        restored_raw.quantum_weights.dtype,
    )
    print(
        "EMA quantum dtype        :",
        restored_ema.quantum_weights.dtype,
    )

    assert payload["epoch"] == 0
    assert (
        payload["model_metadata"]
        ["gate"]["baseline_id"]
        == "B6"
    )

    assert (
        payload["optimizer_state_dict"]
        is not None
    )
    assert len(
        payload["optimizer_state_dict"]["state"]
    ) > 0

    assert (
        raw_comparison["exact"]
        == raw_comparison["total"]
    )
    assert (
        ema_comparison["exact"]
        == ema_comparison["total"]
    )
    assert (
        raw_comparison["max_difference"]
        == 0.0
    )
    assert (
        ema_comparison["max_difference"]
        == 0.0
    )

    assert (
        restored_raw.quantum_weights.dtype
        == torch.float64
    )
    assert (
        restored_ema.quantum_weights.dtype
        == torch.float64
    )

    report = {
        "last_path": str(trainer.last),
        "best_path": str(trainer.best),
        "last_bytes": trainer.last.stat().st_size,
        "best_bytes": trainer.best.stat().st_size,
        "checkpoint_format": payload["qvf_format"],
        "baseline": payload[
            "model_metadata"
        ]["gate"]["baseline_id"],
        "raw_state": raw_comparison,
        "ema_state": ema_comparison,
        "optimizer_state_entries": len(
            payload[
                "optimizer_state_dict"
            ]["state"]
        ),
        "raw_quantum_dtype": str(
            restored_raw.quantum_weights.dtype
        ),
        "ema_quantum_dtype": str(
            restored_ema.quantum_weights.dtype
        ),
    }

    report_path = (
        OUTPUT_DIRECTORY
        / "quantum_trainer_checkpoint_test.json"
    )

    report_path.write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )

    for item in (
        model,
        trainer.ema.ema,
        restored_raw,
        restored_ema,
    ):
        item.close_gate_hook()

    del trainer
    del model
    del restored_raw
    del restored_ema
    del batch
    del loss_components
    del loss_items
    del loss

    gc.collect()
    torch.cuda.empty_cache()

    print("Report                   :", report_path)
    print()
    print(
        "QUANTUM TRAINER CHECKPOINT TEST: PASSED"
    )


if __name__ == "__main__":
    main()
