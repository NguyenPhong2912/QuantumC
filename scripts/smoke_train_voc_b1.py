from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from ultralytics.utils.torch_utils import unwrap_model

from src.qvisionframe.detection_trainer import (
    QVisionDetectionTrainer,
)
from src.qvisionframe.gated_detection_model import (
    GatedDetectionModel,
)


SEED = 42


def set_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def clone_gate_state(
    model: torch.nn.Module,
) -> dict[str, torch.Tensor]:
    model = unwrap_model(model)

    if not isinstance(model, GatedDetectionModel):
        raise TypeError(
            "Expected GatedDetectionModel, received "
            f"{type(model).__name__}"
        )

    if model.gate is None:
        raise RuntimeError("B1 gate is missing")

    return {
        key: value.detach().cpu().clone()
        for key, value in model.gate.state_dict().items()
    }


def compare_states(
    before: dict[str, torch.Tensor],
    after: dict[str, torch.Tensor],
) -> dict[str, dict[str, float | bool]]:
    if before.keys() != after.keys():
        raise RuntimeError(
            "Initial and final gate state keys differ"
        )

    comparison: dict[
        str,
        dict[str, float | bool],
    ] = {}

    for key in before:
        difference = (
            after[key].float()
            - before[key].float()
        )

        comparison[key] = {
            "changed": not torch.equal(
                before[key],
                after[key],
            ),
            "max_abs_change": float(
                difference.abs().max()
            ),
            "l2_change": float(
                difference.norm()
            ),
        }

    return comparison


def checkpoint_model(
    path: Path,
) -> GatedDetectionModel:
    checkpoint: Any = torch.load(
        path,
        map_location="cpu",
        weights_only=False,
    )

    if not isinstance(checkpoint, dict):
        raise TypeError(
            f"Unexpected checkpoint: {type(checkpoint)}"
        )

    model = checkpoint.get("ema")

    if model is None:
        model = checkpoint.get("model")

    if model is None:
        raise KeyError(
            "Checkpoint contains neither ema nor model"
        )

    model = model.float()

    if not isinstance(model, GatedDetectionModel):
        raise TypeError(
            "Checkpoint model is not GatedDetectionModel: "
            f"{type(model).__name__}"
        )

    return model


def main() -> None:
    print("===== VOC B1 SMOKE TRAINING =====")

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")

    set_seed(SEED)

    root = Path(__file__).resolve().parents[1]
    server = os.environ.get(
        "QVF_SERVER",
        "FITLAB-01",
    )

    project = (
        root
        / "outputs"
        / server
        / "smoke_training"
    )
    run_name = "voc_b1_linear_seed42"

    observations: dict[str, Any] = {}

    overrides = {
        "model": "yolo11n.yaml",
        "data": str(
            root / "configs/datasets/voc.yaml"
        ),
        "epochs": 1,
        "imgsz": 320,
        "batch": 4,
        "device": 0,
        "workers": 2,
        "fraction": 0.01,
        "pretrained": False,
        "optimizer": "SGD",
        "lr0": 0.01,
        "momentum": 0.937,
        "weight_decay": 0.0005,
        "seed": SEED,
        "deterministic": True,
        "amp": False,
        "val": False,
        "plots": False,
        "save": True,
        "save_period": 1,
        "project": str(project),
        "name": run_name,
        "exist_ok": True,
        "verbose": True,
    }

    trainer = QVisionDetectionTrainer(
        baseline_id="B1",
        target_layer=10,
        gate_channels=256,
        alpha_init=1.0e-3,
        overrides=overrides,
    )

    def on_train_start(current_trainer) -> None:
        model = unwrap_model(current_trainer.model)

        observations["initial_gate_state"] = (
            clone_gate_state(model)
        )
        observations["initial_hook_calls"] = (
            model.hook_calls
        )

        optimizer_parameter_ids = {
            id(parameter)
            for group in current_trainer.optimizer.param_groups
            for parameter in group["params"]
        }

        observations["gate_in_optimizer"] = {
            name: id(parameter)
            in optimizer_parameter_ids
            for name, parameter in model.named_parameters()
            if name.startswith("gate.")
        }

        print()
        print("===== B1 TRAIN START AUDIT =====")
        print(
            "Model type       :",
            type(model).__name__,
        )
        print(
            "Gate in optimizer:",
            observations["gate_in_optimizer"],
        )

    def on_train_end(current_trainer) -> None:
        model = unwrap_model(current_trainer.model)

        observations["final_gate_state"] = (
            clone_gate_state(model)
        )
        observations["final_hook_calls"] = (
            model.hook_calls
        )

    trainer.add_callback(
        "on_train_start",
        on_train_start,
    )
    trainer.add_callback(
        "on_train_end",
        on_train_end,
    )

    trainer.train()

    initial_state = observations.get(
        "initial_gate_state"
    )
    final_state = observations.get(
        "final_gate_state"
    )

    if initial_state is None or final_state is None:
        raise RuntimeError(
            "Training callbacks did not capture gate state"
        )

    state_changes = compare_states(
        initial_state,
        final_state,
    )

    gate_in_optimizer = observations[
        "gate_in_optimizer"
    ]

    save_dir = Path(trainer.save_dir)
    last_path = save_dir / "weights/last.pt"
    best_path = save_dir / "weights/best.pt"
    epoch_path = save_dir / "weights/epoch0.pt"
    results_path = save_dir / "results.csv"
    args_path = save_dir / "args.yaml"

    checkpoint_path = (
        best_path
        if best_path.is_file()
        else last_path
    )

    loaded_model = checkpoint_model(
        checkpoint_path
    )

    checkpoint_gate_keys = [
        key
        for key in loaded_model.state_dict()
        if key.startswith("gate.")
    ]

    checkpoint_gate_state = {
        key: value.detach().cpu()
        for key, value
        in loaded_model.gate.state_dict().items()
    }

    checkpoint_vs_final = compare_states(
        final_state,
        checkpoint_gate_state,
    )

    loaded_model = loaded_model.cuda().eval()
    loaded_model.reset_gate_observations()

    with torch.no_grad():
        inference_output = loaded_model(
            torch.rand(
                1,
                3,
                320,
                320,
                device="cuda:0",
            )
        )

    summary = {
        "server": server,
        "save_dir": str(save_dir),
        "seed": SEED,
        "baseline_id": "B1",
        "fraction": 0.01,
        "epochs": 1,
        "imgsz": 320,
        "batch": 4,
        "gate_in_optimizer": gate_in_optimizer,
        "gate_state_changes": state_changes,
        "initial_hook_calls": observations[
            "initial_hook_calls"
        ],
        "final_hook_calls": observations[
            "final_hook_calls"
        ],
        "checkpoint": str(checkpoint_path),
        "checkpoint_gate_keys": (
            checkpoint_gate_keys
        ),
        "checkpoint_vs_final": (
            checkpoint_vs_final
        ),
        "reload_feature_shape": (
            loaded_model.last_feature_shape
        ),
        "reload_hook_calls": (
            loaded_model.hook_calls
        ),
        "inference_output_type": (
            type(inference_output).__name__
        ),
        "last_exists": last_path.is_file(),
        "best_exists": best_path.is_file(),
        "epoch0_exists": epoch_path.is_file(),
        "results_exists": results_path.is_file(),
        "args_exists": args_path.is_file(),
    }

    summary_path = save_dir / "b1_smoke_summary.json"
    summary_path.write_text(
        json.dumps(
            summary,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    print()
    print("===== B1 GATE UPDATE AUDIT =====")
    print("Save directory       :", save_dir)
    print("Gate in optimizer    :", gate_in_optimizer)

    for name, values in state_changes.items():
        print(
            f"{name:<14} | "
            f"changed={values['changed']} | "
            f"max={values['max_abs_change']:.12g} | "
            f"l2={values['l2_change']:.12g}"
        )

    print("Final hook calls     :", observations["final_hook_calls"])
    print("Checkpoint used      :", checkpoint_path)
    print("Checkpoint gate keys :", checkpoint_gate_keys)
    print("Reload feature shape :", loaded_model.last_feature_shape)
    print("Reload hook calls    :", loaded_model.hook_calls)
    print("Summary              :", summary_path)

    assert all(gate_in_optimizer.values())
    assert set(gate_in_optimizer) == {
        "gate.alpha",
        "gate.linear.weight",
        "gate.linear.bias",
    }

    # At minimum, alpha and linear weights must update.
    assert state_changes["alpha"]["changed"]
    assert state_changes["linear.weight"]["changed"]
    assert (
        state_changes["alpha"]["max_abs_change"]
        > 0.0
    )
    assert (
        state_changes["linear.weight"][
            "max_abs_change"
        ]
        > 0.0
    )

    assert checkpoint_gate_keys == [
        "gate.alpha",
        "gate.linear.weight",
        "gate.linear.bias",
    ]

    assert loaded_model.last_feature_shape == (
        1,
        256,
        10,
        10,
    )
    assert loaded_model.hook_calls == 1

    assert last_path.is_file()
    assert results_path.is_file()
    assert args_path.is_file()

    loaded_model.close_gate_hook()

    print("VOC B1 SMOKE TRAINING: PASSED")


if __name__ == "__main__":
    main()
