from __future__ import annotations

import argparse
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

BASELINE_NAMES = {
    "B1": "linear",
    "B2": "channel_attention",
    "B3": "classical_mlp",
    "B4": "parameter_matched",
    "B5": "trigonometric",
}

EXPECTED_PARAMETERS = {
    "B1": 2725069,
    "B2": 2602205,
    "B3": 2628189,
    "B4": 2628129,
    "B5": 2628153,
}

EXPECTED_GATE_STATE_KEYS = {
    "B1": 3,
    "B2": 5,
    "B3": 11,
    "B4": 11,
    "B5": 11,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a one-epoch VOC smoke test for a "
            "Q-VisionFrame classical baseline."
        )
    )

    parser.add_argument(
        "--baseline",
        required=True,
        choices=tuple(BASELINE_NAMES),
    )

    return parser.parse_args()


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
        raise RuntimeError("Feature gate is missing")

    return {
        key: tensor.detach().cpu().clone()
        for key, tensor
        in model.gate.state_dict().items()
    }


def state_changes(
    before: dict[str, torch.Tensor],
    after: dict[str, torch.Tensor],
) -> dict[str, dict[str, float | bool]]:
    if before.keys() != after.keys():
        raise RuntimeError(
            "Gate state keys changed during training"
        )

    output: dict[
        str,
        dict[str, float | bool],
    ] = {}

    for key in before:
        difference = (
            after[key].float()
            - before[key].float()
        )

        output[key] = {
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

    return output


def fp16_serialization_roundtrip(
    state: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """
    Reproduce Ultralytics checkpoint model serialization:

        deepcopy(ema).half()

    Checkpoints are later loaded and converted back to float32,
    so floating tensors must be compared against
    tensor.half().float(), not the original FP32 EMA tensor.
    """
    output: dict[str, torch.Tensor] = {}

    for key, tensor in state.items():
        value = tensor.detach().cpu().clone()

        if value.is_floating_point():
            value = value.half().float()

        output[key] = value

    return output


def count_parameters(
    model: torch.nn.Module,
) -> int:
    return sum(
        parameter.numel()
        for parameter in model.parameters()
    )


def load_checkpoint_model(
    checkpoint_path: Path,
) -> GatedDetectionModel:
    checkpoint: Any = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )

    if not isinstance(checkpoint, dict):
        raise TypeError(
            "Expected checkpoint dictionary, received "
            f"{type(checkpoint).__name__}"
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
    cli = parse_args()
    baseline_id: str = cli.baseline

    print(
        f"===== VOC {baseline_id} "
        "CLASSICAL SMOKE TRAINING ====="
    )

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")

    set_seed(SEED)

    root = Path(__file__).resolve().parents[1]

    server = os.environ.get(
        "QVF_SERVER",
        "FITLAB-01",
    )

    baseline_name = BASELINE_NAMES[baseline_id]

    run_name = (
        f"voc_{baseline_id.lower()}_"
        f"{baseline_name}_seed42"
    )

    project = (
        root
        / "outputs"
        / server
        / "smoke_training"
    )

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
        baseline_id=baseline_id,
        target_layer=10,
        gate_channels=256,
        latent_dim=4,
        hidden_dim=64,
        transform_hidden_dim=8,
        n_layers=2,
        reduction=16,
        alpha_init=1.0e-3,
        overrides=overrides,
    )

    def on_train_start(
        current_trainer,
    ) -> None:
        model = unwrap_model(
            current_trainer.model
        )

        if not isinstance(
            model,
            GatedDetectionModel,
        ):
            raise TypeError(
                type(model).__name__
            )

        observations["initial_state"] = (
            clone_gate_state(model)
        )

        observations["initial_hook_calls"] = (
            model.hook_calls
        )

        optimizer_parameter_ids = {
            id(parameter)
            for group
            in current_trainer.optimizer.param_groups
            for parameter in group["params"]
        }

        observations["gate_in_optimizer"] = {
            name: (
                id(parameter)
                in optimizer_parameter_ids
            )
            for name, parameter
            in model.named_parameters()
            if name.startswith("gate.")
        }

        observations["total_parameters"] = (
            count_parameters(model)
        )

        print()
        print("===== TRAIN START AUDIT =====")
        print("Baseline          :", baseline_id)
        print(
            "Model type        :",
            type(model).__name__,
        )
        print(
            "Gate type         :",
            type(model.gate).__name__,
        )
        print(
            "Total parameters  :",
            observations["total_parameters"],
        )
        print(
            "Gate in optimizer :",
            observations["gate_in_optimizer"],
        )

    def on_train_end(
        current_trainer,
    ) -> None:
        model = unwrap_model(
            current_trainer.model
        )

        observations["final_raw_state"] = (
            clone_gate_state(model)
        )

        observations["final_hook_calls"] = (
            model.hook_calls
        )

        if current_trainer.ema is None:
            raise RuntimeError(
                "Trainer EMA is unavailable"
            )

        ema_model = unwrap_model(
            current_trainer.ema.ema
        )

        observations["final_ema_state"] = (
            clone_gate_state(ema_model)
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
        "initial_state"
    )

    final_raw_state = observations.get(
        "final_raw_state"
    )

    final_ema_state = observations.get(
        "final_ema_state"
    )

    if initial_state is None:
        raise RuntimeError(
            "Initial gate state was not captured"
        )

    if final_raw_state is None:
        raise RuntimeError(
            "Final raw gate state was not captured"
        )

    if final_ema_state is None:
        raise RuntimeError(
            "Final EMA gate state was not captured"
        )

    changes = state_changes(
        initial_state,
        final_raw_state,
    )

    raw_vs_ema = state_changes(
        final_raw_state,
        final_ema_state,
    )

    save_dir = Path(trainer.save_dir)

    last_path = save_dir / "weights/last.pt"
    best_path = save_dir / "weights/best.pt"
    results_path = save_dir / "results.csv"
    args_path = save_dir / "args.yaml"

    checkpoint_path = (
        best_path
        if best_path.is_file()
        else last_path
    )

    loaded_model = load_checkpoint_model(
        checkpoint_path
    )

    checkpoint_gate_state = {
        key: tensor.detach().cpu().clone()
        for key, tensor
        in loaded_model.gate.state_dict().items()
    }

    checkpoint_state_keys = [
        key
        for key in loaded_model.state_dict()
        if key.startswith("gate.")
    ]

    serialized_ema_state = (
        fp16_serialization_roundtrip(
            final_ema_state
        )
    )

    checkpoint_vs_serialized_ema = state_changes(
        serialized_ema_state,
        checkpoint_gate_state,
    )

    checkpoint_vs_fp32_ema = state_changes(
        final_ema_state,
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

    changed_trainable_parameters = {
        name: values["changed"]
        for name, values in changes.items()
        if name in dict(
            loaded_model.gate.named_parameters()
        )
    }

    summary = {
        "server": server,
        "baseline_id": baseline_id,
        "baseline_name": baseline_name,
        "seed": SEED,
        "save_dir": str(save_dir),
        "total_parameters": observations[
            "total_parameters"
        ],
        "gate_type": type(
            loaded_model.gate
        ).__name__,
        "gate_in_optimizer": observations[
            "gate_in_optimizer"
        ],
        "gate_state_changes": changes,
        "changed_trainable_parameters": (
            changed_trainable_parameters
        ),
        "initial_hook_calls": observations[
            "initial_hook_calls"
        ],
        "final_hook_calls": observations[
            "final_hook_calls"
        ],
        "checkpoint": str(checkpoint_path),
        "checkpoint_gate_keys": (
            checkpoint_state_keys
        ),
        "raw_vs_ema": raw_vs_ema,
        "checkpoint_vs_fp32_ema": (
            checkpoint_vs_fp32_ema
        ),
        "checkpoint_vs_serialized_ema": (
            checkpoint_vs_serialized_ema
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
        "results_exists": results_path.is_file(),
        "args_exists": args_path.is_file(),
    }

    summary_path = (
        save_dir
        / f"{baseline_id.lower()}_smoke_summary.json"
    )

    summary_path.write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )

    print()
    print("===== GATE UPDATE AUDIT =====")
    print("Baseline             :", baseline_id)
    print(
        "Gate type            :",
        type(loaded_model.gate).__name__,
    )
    print(
        "Gate in optimizer    :",
        observations["gate_in_optimizer"],
    )

    for name, values in changes.items():
        print(
            f"{name:<30} | "
            f"changed={values['changed']} | "
            f"max={values['max_abs_change']:.12g} | "
            f"l2={values['l2_change']:.12g}"
        )

    print()
    print("===== RAW MODEL VS EMA =====")

    for name, values in raw_vs_ema.items():
        print(
            f"{name:<30} | "
            f"changed={values['changed']} | "
            f"max={values['max_abs_change']:.12g} | "
            f"l2={values['l2_change']:.12g}"
        )

    print(
        "Final hook calls     :",
        observations["final_hook_calls"],
    )
    print()
    print("===== CHECKPOINT VS FP32 EMA =====")

    for name, values in checkpoint_vs_fp32_ema.items():
        print(
            f"{name:<30} | "
            f"changed={values['changed']} | "
            f"max={values['max_abs_change']:.12g} | "
            f"l2={values['l2_change']:.12g}"
        )

    print()
    print("===== CHECKPOINT VS SERIALIZED EMA =====")

    for name, values in (
        checkpoint_vs_serialized_ema.items()
    ):
        print(
            f"{name:<30} | "
            f"changed={values['changed']} | "
            f"max={values['max_abs_change']:.12g} | "
            f"l2={values['l2_change']:.12g}"
        )

    print(
        "Checkpoint gate keys :",
        len(checkpoint_state_keys),
    )
    print(
        "Reload feature shape :",
        loaded_model.last_feature_shape,
    )
    print(
        "Reload hook calls    :",
        loaded_model.hook_calls,
    )
    print("Summary              :", summary_path)

    gate_in_optimizer = observations[
        "gate_in_optimizer"
    ]

    assert gate_in_optimizer
    assert all(gate_in_optimizer.values())

    assert (
        observations["total_parameters"]
        == EXPECTED_PARAMETERS[baseline_id]
    )

    assert (
        len(checkpoint_state_keys)
        == EXPECTED_GATE_STATE_KEYS[baseline_id]
    )

    assert changed_trainable_parameters
    assert all(
        changed_trainable_parameters.values()
    )

    # Ultralytics serializes the EMA model in FP16.
    # Therefore the loaded FP32 checkpoint must match
    # EMA.half().float(), not the original FP32 EMA.
    assert all(
        not values["changed"]
        for values
        in checkpoint_vs_serialized_ema.values()
    )

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

    print(
        f"VOC {baseline_id} "
        "CLASSICAL SMOKE TRAINING: PASSED"
    )


if __name__ == "__main__":
    main()
