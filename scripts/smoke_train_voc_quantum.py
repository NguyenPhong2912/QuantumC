from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import torch

from src.qvisionframe.quantum_checkpoint import (
    load_quantum_checkpoint,
    reconstruct_quantum_model,
)
from src.qvisionframe.quantum_detection_trainer import (
    QuantumDetectionTrainer,
)


ROOT = Path(__file__).resolve().parents[1]


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--baseline",
        choices=("B6", "B7", "B8"),
        default="B6",
    )
    parser.add_argument(
        "--device",
        default="0",
    )

    return parser.parse_args()


def state_inventory(
    state: dict[str, torch.Tensor],
) -> dict[str, Any]:
    gate_keys = [
        key
        for key in state
        if key.startswith("gate.")
    ]

    quantum_tensor = state[
        "gate.quantum_weights"
    ]

    return {
        "state_tensors": len(state),
        "gate_state_keys": len(gate_keys),
        "quantum_dtype": str(
            quantum_tensor.dtype
        ),
        "quantum_device": str(
            quantum_tensor.device
        ),
        "finite": all(
            bool(torch.isfinite(value).all())
            for value in state.values()
            if value.is_floating_point()
        ),
    }


def main() -> None:
    arguments = parse_arguments()
    baseline_id = arguments.baseline

    server = os.environ.get(
        "QVF_SERVER",
        "UNKNOWN",
    )

    run_name = (
        f"voc_{baseline_id.lower()}_"
        "quantum_smoke_seed42"
    )

    project = (
        ROOT
        / "outputs"
        / server
        / "smoke_training"
    )

    print(
        f"===== VOC {baseline_id} "
        "QUANTUM SMOKE TRAINING ====="
    )
    print("Server              :", server)
    print("CUDA available      :", torch.cuda.is_available())

    trainer = QuantumDetectionTrainer(
        quantum_baseline_id=baseline_id,
        quantum_target_layer=10,
        quantum_channels=256,
        quantum_n_qubits=4,
        quantum_n_layers=2,
        quantum_hidden_dim=64,
        quantum_alpha_init=1.0e-3,
        quantum_gamma=0.5,
        overrides={
            "model": "yolo11n.yaml",
            "data": str(
                ROOT
                / "configs/datasets/"
                "voc_smoke_32.yaml"
            ),
            "epochs": 1,
            "imgsz": 320,
            "batch": 2,
            "nbs": 2,
            "device": arguments.device,
            "workers": 2,
            "optimizer": "SGD",
            "lr0": 0.01,
            "lrf": 0.01,
            "momentum": 0.937,
            "weight_decay": 0.0005,
            "warmup_epochs": 0.0,
            "pretrained": False,
            "amp": False,
            "deterministic": True,
            "seed": 42,
            "val": True,
            "plots": False,
            "save": True,
            "save_period": 1,
            "project": str(project),
            "name": run_name,
            "exist_ok": True,
            "verbose": True,
        },
    )

    trainer.train()

    raw_model = trainer.model

    if hasattr(raw_model, "module"):
        raw_model = raw_model.module

    quantum_parameter = (
        raw_model.gate.quantum_weights
    )

    initial_quantum_weights = (
        trainer.qvf_initial_quantum_weights
    )

    final_quantum_weights = (
        quantum_parameter.detach()
        .cpu()
    )

    quantum_delta = (
        final_quantum_weights
        - initial_quantum_weights
    )

    quantum_max_abs_delta = float(
        quantum_delta.abs().max()
    )

    quantum_l2_delta = float(
        torch.linalg.vector_norm(
            quantum_delta
        )
    )

    quantum_in_optimizer = any(
        parameter is quantum_parameter
        for group in trainer.optimizer.param_groups
        for parameter in group["params"]
    )

    print()
    print("===== OPTIMIZER CONTRACT AUDIT =====")
    print(
        "Quantum requires_grad:",
        quantum_parameter.requires_grad,
    )
    print(
        "Quantum in optimizer :",
        quantum_in_optimizer,
    )
    print(
        "Optimizer audit      :",
        trainer.qvf_optimizer_audit,
    )
    print(
        "Quantum max abs delta:",
        quantum_max_abs_delta,
    )
    print(
        "Quantum L2 delta     :",
        quantum_l2_delta,
    )

    if baseline_id == "B6":
        assert not quantum_parameter.requires_grad
        assert not quantum_in_optimizer
        assert (
            trainer.qvf_optimizer_audit[
                "removed_parameter_tensors"
            ]
            == 1
        )
        assert quantum_max_abs_delta == 0.0
        assert quantum_l2_delta == 0.0
    else:
        assert quantum_parameter.requires_grad
        assert quantum_in_optimizer
        assert quantum_max_abs_delta > 0.0
        assert quantum_l2_delta > 0.0

    save_directory = project / run_name
    last_path = save_directory / "weights/last.pt"
    best_path = save_directory / "weights/best.pt"

    print()
    print("===== CHECKPOINT AUDIT =====")
    print("last.pt exists      :", last_path.exists())
    print("best.pt exists      :", best_path.exists())

    assert last_path.exists()
    assert best_path.exists()

    last_payload = load_quantum_checkpoint(
        last_path,
        map_location="cpu",
    )
    best_payload = load_quantum_checkpoint(
        best_path,
        map_location="cpu",
    )

    raw_inventory = state_inventory(
        last_payload["model_state_dict"]
    )
    ema_inventory = state_inventory(
        last_payload["ema_state_dict"]
    )

    optimizer_state = last_payload[
        "optimizer_state_dict"
    ]

    optimizer_entries = (
        len(optimizer_state["state"])
        if optimizer_state is not None
        else 0
    )

    print(
        "Checkpoint format   :",
        last_payload["qvf_format"],
    )
    print(
        "Checkpoint baseline :",
        last_payload[
            "model_metadata"
        ]["gate"]["baseline_id"],
    )
    print(
        "Checkpoint entangle :",
        last_payload[
            "model_metadata"
        ]["gate"]["entanglement"],
    )
    print(
        "Checkpoint epoch    :",
        last_payload["epoch"],
    )
    print(
        "EMA updates         :",
        last_payload["ema_updates"],
    )
    print(
        "Optimizer present   :",
        optimizer_state is not None,
    )
    print(
        "Optimizer entries   :",
        optimizer_entries,
    )
    print(
        "Raw state inventory :",
        raw_inventory,
    )
    print(
        "EMA state inventory :",
        ema_inventory,
    )

    assert (
        last_payload["qvf_format"]
        == "qvisionframe.quantum.v1"
    )
    assert (
        best_payload["qvf_format"]
        == "qvisionframe.quantum.v1"
    )
    assert (
        last_payload[
            "model_metadata"
        ]["gate"]["baseline_id"]
        == baseline_id
    )

    expected_entanglement = {
        "B6": "ring",
        "B7": "none",
        "B8": "ring",
    }[baseline_id]

    checkpoint_entanglement = (
        last_payload[
            "model_metadata"
        ]["gate"]["entanglement"]
    )

    assert (
        checkpoint_entanglement
        == expected_entanglement
    )

    assert (
        best_payload[
            "model_metadata"
        ]["gate"]["entanglement"]
        == expected_entanglement
    )

    assert last_payload["epoch"] == 0
    assert optimizer_state is not None
    assert optimizer_entries > 0

    assert raw_inventory["state_tensors"] == 507
    assert ema_inventory["state_tensors"] == 507
    assert raw_inventory["gate_state_keys"] == 8
    assert ema_inventory["gate_state_keys"] == 8

    assert (
        raw_inventory["quantum_dtype"]
        == "torch.float64"
    )
    assert (
        ema_inventory["quantum_dtype"]
        == "torch.float64"
    )
    assert raw_inventory["finite"]
    assert ema_inventory["finite"]

    restored_ema = reconstruct_quantum_model(
        best_payload,
        state_key="ema_state_dict",
        device="cuda:0",
        strict=True,
        verbose=False,
    ).eval()

    restored_ema.reset_gate_observations()

    with torch.no_grad():
        output = restored_ema(
            torch.randn(
                1,
                3,
                320,
                320,
                dtype=torch.float32,
                device="cuda:0",
            )
        )

    predictions = (
        output[0]
        if isinstance(output, tuple)
        else output
    )

    print()
    print("===== RELOAD AUDIT =====")
    print(
        "Reload model type    :",
        type(restored_ema).__name__,
    )
    print(
        "Reload quantum dtype :",
        restored_ema.quantum_weights.dtype,
    )
    print(
        "Reload quantum device:",
        restored_ema.quantum_weights.device,
    )
    print(
        "Reload hook calls     :",
        restored_ema.hook_calls,
    )
    print(
        "Reload feature shape  :",
        restored_ema.last_feature_shape,
    )
    print(
        "Prediction finite     :",
        bool(torch.isfinite(predictions).all()),
    )

    assert (
        restored_ema.quantum_weights.dtype
        == torch.float64
    )
    assert (
        restored_ema.quantum_weights.device.type
        == "cuda"
    )
    assert restored_ema.hook_calls == 1
    assert (
        restored_ema.last_feature_shape
        == (1, 256, 10, 10)
    )
    assert bool(
        torch.isfinite(predictions).all()
    )

    summary = {
        "server": server,
        "baseline": baseline_id,
        "entanglement": (
            checkpoint_entanglement
        ),
        "save_directory": str(
            save_directory
        ),
        "last_path": str(last_path),
        "best_path": str(best_path),
        "last_bytes": last_path.stat().st_size,
        "best_bytes": best_path.stat().st_size,
        "epoch": last_payload["epoch"],
        "ema_updates": (
            last_payload["ema_updates"]
        ),
        "optimizer_state_entries": (
            optimizer_entries
        ),
        "quantum_max_abs_delta": (
            quantum_max_abs_delta
        ),
        "quantum_l2_delta": (
            quantum_l2_delta
        ),
        "raw_state": raw_inventory,
        "ema_state": ema_inventory,
        "reload_quantum_dtype": str(
            restored_ema.quantum_weights.dtype
        ),
        "reload_hook_calls": (
            restored_ema.hook_calls
        ),
        "reload_feature_shape": list(
            restored_ema.last_feature_shape
        ),
    }

    summary_path = (
        save_directory
        / f"{baseline_id.lower()}_"
        "quantum_smoke_summary.json"
    )

    summary_path.write_text(
        json.dumps(
            summary,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    restored_ema.close_gate_hook()

    print("Summary              :", summary_path)
    print()
    print(
        f"VOC {baseline_id} QUANTUM "
        "SMOKE TRAINING: PASSED"
    )


if __name__ == "__main__":
    main()
