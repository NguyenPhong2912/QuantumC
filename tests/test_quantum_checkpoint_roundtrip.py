from __future__ import annotations

import gc
import json
from pathlib import Path
from typing import Any

import torch
from ultralytics.cfg import get_cfg
from ultralytics.utils import DEFAULT_CFG
from ultralytics.utils.torch_utils import ModelEMA

from src.qvisionframe.quantum_checkpoint import (
    QVF_QUANTUM_FORMAT,
    build_quantum_checkpoint,
    load_quantum_checkpoint,
    reconstruct_quantum_model,
    save_quantum_checkpoint,
)
from src.qvisionframe.quantum_detection_model import (
    QuantumGatedDetectionModel,
)


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_PATH = (
    ROOT
    / "outputs/inventory/"
    "quantum_checkpoint_roundtrip.pt"
)
REPORT_PATH = (
    ROOT
    / "outputs/inventory/"
    "quantum_checkpoint_roundtrip.json"
)


def make_model() -> QuantumGatedDetectionModel:
    torch.manual_seed(42)

    model = QuantumGatedDetectionModel(
        cfg="yolo11n.yaml",
        nc=20,
        baseline_id="B8",
        target_layer=10,
        channels=256,
        n_qubits=4,
        n_layers=2,
        hidden_dim=64,
        alpha_init=1.0e-3,
        gamma=0.5,
        verbose=False,
    )

    model.args = get_cfg(cfg=DEFAULT_CFG)
    model.criterion = None

    return model


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
        for key, tensor in model.state_dict().items()
    }


def compare_states(
    first: dict[str, torch.Tensor],
    second: dict[str, torch.Tensor],
) -> dict[str, Any]:
    if first.keys() != second.keys():
        raise RuntimeError(
            "State dictionary keys differ"
        )

    exact = 0
    max_difference = 0.0
    nonexact: list[str] = []

    for key in first:
        tensor_01 = first[key]
        tensor_02 = second[key]

        if (
            tensor_01.dtype == tensor_02.dtype
            and torch.equal(tensor_01, tensor_02)
        ):
            exact += 1
        else:
            nonexact.append(key)

        difference = (
            tensor_01.float()
            - tensor_02.float()
        )

        if difference.numel():
            max_difference = max(
                max_difference,
                float(difference.abs().max()),
            )

    return {
        "exact": exact,
        "total": len(first),
        "max_abs_difference": max_difference,
        "nonexact_keys": nonexact,
    }


def tensor_tree_dtypes(
    value: Any,
    prefix: str = "",
) -> dict[str, str]:
    output: dict[str, str] = {}

    if isinstance(value, torch.Tensor):
        output[prefix] = str(value.dtype)
        return output

    if isinstance(value, dict):
        for key, item in value.items():
            output.update(
                tensor_tree_dtypes(
                    item,
                    f"{prefix}.{key}" if prefix else str(key),
                )
            )

    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            output.update(
                tensor_tree_dtypes(
                    item,
                    f"{prefix}[{index}]",
                )
            )

    return output


def main() -> None:
    print(
        "===== QUANTUM CHECKPOINT ROUNDTRIP ====="
    )

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is unavailable"
        )

    device = torch.device("cuda:0")

    model = make_model().to(device).train()

    # EMA must be created before the raw model's first forward,
    # matching the Ultralytics lifecycle.
    ema = ModelEMA(model)

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=1.0e-3,
        momentum=0.9,
        weight_decay=5.0e-4,
    )

    batch = make_batch(device)

    optimizer.zero_grad(set_to_none=True)

    loss_components, loss_items = model(batch)
    loss = loss_components.sum()
    loss.backward()
    optimizer.step()

    ema.update(model)

    # Materialize the EMA quantum simulator. Whole-object pickle would
    # fail after this point, while state_dict checkpointing must pass.
    ema.ema.eval()

    with torch.no_grad():
        ema_output_before = ema.ema(
            batch["img"]
        )

    raw_state_before = clone_state(model)
    ema_state_before = clone_state(ema.ema)

    raw_quantum_dtype_before = str(
        model.quantum_weights.dtype
    )
    ema_quantum_dtype_before = str(
        ema.ema.quantum_weights.dtype
    )

    optimizer_state_before = optimizer.state_dict()

    payload = build_quantum_checkpoint(
        model=model,
        ema_model=ema.ema,
        ema_updates=ema.updates,
        optimizer=optimizer,
        scaler_state_dict={},
        epoch=0,
        best_fitness=0.125,
        train_args={
            "model": "yolo11n.yaml",
            "data": "configs/datasets/voc.yaml",
            "epochs": 1,
            "imgsz": 320,
            "batch": 2,
            "device": 0,
            "baseline_id": "B8",
        },
        train_metrics={
            "metrics/mAP50(B)": 0.0,
            "fitness": 0.125,
        },
        train_results=[],
        extra_metadata={
            "server": "FITLAB-01",
            "purpose": "isolated-roundtrip-test",
        },
    )

    output_path = save_quantum_checkpoint(
        payload,
        OUTPUT_PATH,
    )

    loaded = load_quantum_checkpoint(
        output_path,
        map_location="cpu",
    )

    restored_raw = reconstruct_quantum_model(
        loaded,
        state_key="model_state_dict",
        device=device,
        strict=True,
        verbose=False,
    ).train()

    restored_ema = reconstruct_quantum_model(
        loaded,
        state_key="ema_state_dict",
        device=device,
        strict=True,
        verbose=False,
    ).eval()

    restored_raw.args = get_cfg(cfg=DEFAULT_CFG)
    restored_raw.criterion = None

    raw_state_after = clone_state(restored_raw)
    ema_state_after = clone_state(restored_ema)

    raw_comparison = compare_states(
        raw_state_before,
        raw_state_after,
    )

    ema_comparison = compare_states(
        ema_state_before,
        ema_state_after,
    )

    # Recreate optimizer with identical parameter ordering before
    # loading its state dictionary.
    restored_optimizer = torch.optim.SGD(
        restored_raw.parameters(),
        lr=1.0e-3,
        momentum=0.9,
        weight_decay=5.0e-4,
    )

    restored_optimizer.load_state_dict(
        loaded["optimizer_state_dict"]
    )

    optimizer_state_after = (
        restored_optimizer.state_dict()
    )

    optimizer_before_dtypes = tensor_tree_dtypes(
        optimizer_state_before
    )
    optimizer_after_dtypes = tensor_tree_dtypes(
        optimizer_state_after
    )

    with torch.no_grad():
        ema_output_after = restored_ema(
            batch["img"]
        )

    if isinstance(ema_output_before, tuple):
        prediction_before = (
            ema_output_before[0]
            .detach()
            .float()
            .cpu()
        )
    else:
        prediction_before = (
            ema_output_before
            .detach()
            .float()
            .cpu()
        )

    if isinstance(ema_output_after, tuple):
        prediction_after = (
            ema_output_after[0]
            .detach()
            .float()
            .cpu()
        )
    else:
        prediction_after = (
            ema_output_after
            .detach()
            .float()
            .cpu()
        )

    output_exact = torch.equal(
        prediction_before,
        prediction_after,
    )

    output_max_difference = float(
        (
            prediction_before
            - prediction_after
        )
        .abs()
        .max()
    )

    print()
    print("Checkpoint format       :", loaded["qvf_format"])
    print("Checkpoint bytes        :", output_path.stat().st_size)
    print("Epoch                   :", loaded["epoch"])
    print("EMA updates             :", loaded["ema_updates"])
    print("Raw exact state tensors :", f"{raw_comparison['exact']}/{raw_comparison['total']}")
    print("EMA exact state tensors :", f"{ema_comparison['exact']}/{ema_comparison['total']}")
    print("Raw max state difference:", raw_comparison["max_abs_difference"])
    print("EMA max state difference:", ema_comparison["max_abs_difference"])
    print("EMA output exact        :", output_exact)
    print("EMA max output diff     :", output_max_difference)
    print("Raw quantum dtype before:", raw_quantum_dtype_before)
    print("Raw quantum dtype after :", restored_raw.quantum_weights.dtype)
    print("EMA quantum dtype before:", ema_quantum_dtype_before)
    print("EMA quantum dtype after :", restored_ema.quantum_weights.dtype)
    print("Optimizer state entries :", len(optimizer_state_after["state"]))
    print("Optimizer dtypes equal  :", optimizer_before_dtypes == optimizer_after_dtypes)

    assert loaded["qvf_format"] == QVF_QUANTUM_FORMAT
    assert loaded["format_version"] == 1
    assert loaded["epoch"] == 0
    assert loaded["ema_updates"] == ema.updates

    assert (
        raw_comparison["exact"]
        == raw_comparison["total"]
    )
    assert (
        ema_comparison["exact"]
        == ema_comparison["total"]
    )
    assert (
        raw_comparison["max_abs_difference"]
        == 0.0
    )
    assert (
        ema_comparison["max_abs_difference"]
        == 0.0
    )

    assert output_exact
    assert output_max_difference == 0.0

    assert (
        model.quantum_weights.dtype
        == torch.float64
    )
    assert (
        ema.ema.quantum_weights.dtype
        == torch.float64
    )
    assert (
        restored_raw.quantum_weights.dtype
        == torch.float64
    )
    assert (
        restored_ema.quantum_weights.dtype
        == torch.float64
    )

    assert optimizer_before_dtypes == (
        optimizer_after_dtypes
    )

    assert len(
        optimizer_state_after["state"]
    ) > 0

    report = {
        "checkpoint_path": str(output_path),
        "checkpoint_bytes": (
            output_path.stat().st_size
        ),
        "format": loaded["qvf_format"],
        "epoch": loaded["epoch"],
        "ema_updates": loaded["ema_updates"],
        "raw_state_comparison": raw_comparison,
        "ema_state_comparison": ema_comparison,
        "ema_output_exact": output_exact,
        "ema_max_output_difference": (
            output_max_difference
        ),
        "raw_quantum_dtype_before": (
            raw_quantum_dtype_before
        ),
        "raw_quantum_dtype_after": str(
            restored_raw.quantum_weights.dtype
        ),
        "ema_quantum_dtype_before": (
            ema_quantum_dtype_before
        ),
        "ema_quantum_dtype_after": str(
            restored_ema.quantum_weights.dtype
        ),
        "optimizer_state_entries": len(
            optimizer_state_after["state"]
        ),
        "optimizer_dtypes_equal": (
            optimizer_before_dtypes
            == optimizer_after_dtypes
        ),
    }

    REPORT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    REPORT_PATH.write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )

    for item in (
        model,
        ema.ema,
        restored_raw,
        restored_ema,
    ):
        item.close_gate_hook()

    del model
    del ema
    del restored_raw
    del restored_ema
    del optimizer
    del restored_optimizer
    del batch
    del loss_components
    del loss_items
    del loss
    del ema_output_before
    del ema_output_after

    gc.collect()
    torch.cuda.empty_cache()

    print("Report                  :", REPORT_PATH)
    print()
    print(
        "QUANTUM CHECKPOINT ROUNDTRIP: PASSED"
    )


if __name__ == "__main__":
    main()
