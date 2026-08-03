from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch

from src.qvisionframe.quantum_checkpoint import (
    load_quantum_checkpoint,
)
from src.qvisionframe.quantum_detection_trainer import (
    QuantumDetectionTrainer,
)


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--baseline",
        choices=("B6", "B7", "B8"),
        default="B8",
    )
    parser.add_argument(
        "--device",
        default="0",
    )
    parser.add_argument(
        "--source-server",
        default=None,
        help=(
            "Server namespace containing the source "
            "smoke checkpoint. Defaults to QVF_SERVER."
        ),
    )

    return parser.parse_args()


def main() -> None:
    arguments = parse_args()
    baseline = arguments.baseline

    server = os.environ.get(
        "QVF_SERVER",
        "UNKNOWN",
    )

    source_server = (
        arguments.source_server
        if arguments.source_server is not None
        else server
    )

    source_run = (
        ROOT
        / "outputs"
        / source_server
        / "smoke_training"
        / f"voc_{baseline.lower()}_"
        "quantum_smoke_seed42"
    )

    source_checkpoint = (
        source_run / "weights/last.pt"
    )

    if not source_checkpoint.exists():
        raise FileNotFoundError(
            source_checkpoint
        )

    source_payload = load_quantum_checkpoint(
        source_checkpoint,
        map_location="cpu",
    )

    source_raw_quantum = (
        source_payload["model_state_dict"]
        ["gate.quantum_weights"]
        .clone()
    )

    source_epoch = int(
        source_payload["epoch"]
    )
    source_ema_updates = int(
        source_payload["ema_updates"]
    )
    source_optimizer_entries = len(
        source_payload[
            "optimizer_state_dict"
        ]["state"]
    )

    print(
        f"===== VOC {baseline} "
        "QUANTUM RESUME SMOKE ====="
    )
    print("Current server    :", server)
    print("Source server     :", source_server)
    print("Source checkpoint :", source_checkpoint)
    print("Source epoch      :", source_epoch)
    print("Source EMA updates:", source_ema_updates)
    print(
        "Source optimizer entries:",
        source_optimizer_entries,
    )

    assert source_epoch == 0
    assert source_ema_updates == 16
    assert source_optimizer_entries > 0

    run_name = (
        f"voc_{baseline.lower()}_"
        "quantum_resume_seed42"
    )

    project = (
        ROOT
        / "outputs"
        / server
        / "resume_training"
    )

    trainer = QuantumDetectionTrainer(
        quantum_baseline_id=baseline,
        quantum_target_layer=10,
        quantum_channels=256,
        quantum_n_qubits=4,
        quantum_n_layers=2,
        quantum_hidden_dim=64,
        quantum_alpha_init=1.0e-3,
        quantum_gamma=0.5,
        quantum_resume_path=source_checkpoint,
        overrides={
            "model": "yolo11n.yaml",
            "data": str(
                ROOT
                / "configs/datasets/"
                "voc_smoke_32.yaml"
            ),
            "epochs": 2,
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

    output_checkpoint = (
        project
        / run_name
        / "weights/last.pt"
    )

    assert output_checkpoint.exists()

    resumed_payload = load_quantum_checkpoint(
        output_checkpoint,
        map_location="cpu",
    )

    resumed_raw_quantum = (
        resumed_payload["model_state_dict"]
        ["gate.quantum_weights"]
    )

    quantum_delta = (
        resumed_raw_quantum
        - source_raw_quantum
    )

    max_abs_delta = float(
        quantum_delta.abs().max()
    )
    l2_delta = float(
        torch.linalg.vector_norm(
            quantum_delta
        )
    )

    resumed_epoch = int(
        resumed_payload["epoch"]
    )
    resumed_ema_updates = int(
        resumed_payload["ema_updates"]
    )
    resumed_optimizer_entries = len(
        resumed_payload[
            "optimizer_state_dict"
        ]["state"]
    )

    print()
    print("===== RESUME AUDIT =====")
    print("Resume audit object  :", trainer.qvf_resume_audit)
    print(
        "Loaded optimizer devices:",
        trainer.qvf_resume_audit[
            "optimizer_state_devices"
        ],
    )
    print(
        "Loaded optimizer dtypes :",
        trainer.qvf_resume_audit[
            "optimizer_state_dtypes"
        ],
    )
    print(
        "Loaded optimizer tensors:",
        trainer.qvf_resume_audit[
            "optimizer_state_tensor_count"
        ],
    )
    print("Output epoch         :", resumed_epoch)
    print("Output EMA updates   :", resumed_ema_updates)
    print(
        "Output optimizer entries:",
        resumed_optimizer_entries,
    )
    print("Quantum max abs delta:", max_abs_delta)
    print("Quantum L2 delta     :", l2_delta)
    print(
        "Raw quantum dtype   :",
        resumed_raw_quantum.dtype,
    )
    print(
        "EMA quantum dtype   :",
        resumed_payload[
            "ema_state_dict"
        ]["gate.quantum_weights"].dtype,
    )

    assert (
        trainer.qvf_resume_audit[
            "checkpoint_epoch"
        ]
        == 0
    )
    assert (
        trainer.qvf_resume_audit[
            "ema_updates"
        ]
        == 16
    )
    assert (
        trainer.qvf_resume_audit[
            "optimizer_state_entries"
        ]
        == source_optimizer_entries
    )
    assert (
        trainer.qvf_resume_audit[
            "optimizer_state_tensor_count"
        ]
        == source_optimizer_entries
    )
    assert (
        trainer.qvf_resume_audit[
            "optimizer_state_devices"
        ]
        == ["cuda:0"]
    )
    assert "torch.float32" in (
        trainer.qvf_resume_audit[
            "optimizer_state_dtypes"
        ]
    )

    if baseline != "B6":
        assert "torch.float64" in (
            trainer.qvf_resume_audit[
                "optimizer_state_dtypes"
            ]
        )

    assert resumed_epoch == 1
    assert resumed_ema_updates == 32
    assert resumed_optimizer_entries > 0

    assert resumed_raw_quantum.dtype == torch.float64
    assert (
        resumed_payload[
            "ema_state_dict"
        ]["gate.quantum_weights"].dtype
        == torch.float64
    )

    if baseline == "B6":
        assert max_abs_delta == 0.0
        assert l2_delta == 0.0
    else:
        assert max_abs_delta > 0.0
        assert l2_delta > 0.0

    report = {
        "baseline": baseline,
        "current_server": server,
        "source_server": source_server,
        "source_checkpoint": str(
            source_checkpoint
        ),
        "output_checkpoint": str(
            output_checkpoint
        ),
        "source_epoch": source_epoch,
        "output_epoch": resumed_epoch,
        "source_ema_updates": (
            source_ema_updates
        ),
        "output_ema_updates": (
            resumed_ema_updates
        ),
        "source_optimizer_entries": (
            source_optimizer_entries
        ),
        "output_optimizer_entries": (
            resumed_optimizer_entries
        ),
        "quantum_max_abs_delta": (
            max_abs_delta
        ),
        "quantum_l2_delta": l2_delta,
        "resume_audit": (
            trainer.qvf_resume_audit
        ),
    }

    report_path = (
        project
        / run_name
        / f"{baseline.lower()}_"
        "quantum_resume_summary.json"
    )

    report_path.write_text(
        json.dumps(report, indent=2)
        + "\n",
        encoding="utf-8",
    )

    print("Report               :", report_path)
    print()
    print(
        f"VOC {baseline} QUANTUM "
        "RESUME SMOKE: PASSED"
    )


if __name__ == "__main__":
    main()
