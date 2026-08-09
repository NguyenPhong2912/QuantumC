from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

import torch

from src.qvisionframe.detection_trainer import (
    QVisionDetectionTrainer,
)
from src.qvisionframe.quantum_checkpoint import (
    load_quantum_checkpoint,
)
from src.qvisionframe.quantum_detection_trainer import (
    QuantumDetectionTrainer,
)


ROOT = Path(__file__).resolve().parents[1]

BASELINES = tuple(
    f"B{index}"
    for index in range(9)
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Unified VOC training launcher for "
            "Q-VisionFrame B0-B8."
        )
    )

    parser.add_argument(
        "--baseline",
        required=True,
        choices=BASELINES,
    )
    parser.add_argument(
        "--data-yaml",
        default=str(
            ROOT / "configs/datasets/voc.yaml"
        ),
        help=(
            "Dataset YAML path. Use a host-local override "
            "without modifying the canonical repository YAML."
        ),
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=2,
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
    )
    parser.add_argument(
        "--nbs",
        type=int,
        default=64,
        help="Nominal batch size used for optimizer scaling.",
    )
    parser.add_argument(
        "--cache",
        choices=("false", "ram", "disk"),
        default="false",
    )
    parser.add_argument(
        "--device",
        default="0",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )
    parser.add_argument(
        "--save-period",
        type=int,
        default=10,
    )
    parser.add_argument(
        "--run-suffix",
        default="full",
    )
    parser.add_argument(
        "--resume-from",
        default=None,
        help=(
            "Tensor-only Q-VisionFrame checkpoint used to "
            "resume B6-B8 training."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
    )

    return parser.parse_args()


def build_overrides(
    args: argparse.Namespace,
    *,
    server: str,
    run_name: str,
    data_yaml: Path,
) -> dict[str, Any]:
    project = (
        ROOT
        / "outputs"
        / server
        / "voc_full_training"
    )

    return {
        "model": "yolo11n.yaml",
        "data": str(data_yaml),
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "nbs": args.nbs,
        "device": args.device,
        "workers": args.workers,
        "cache": {
            "false": False,
            "ram": True,
            "disk": "disk",
        }[args.cache],
        "fraction": 1.0,
        "pretrained": False,
        "optimizer": "SGD",
        "lr0": 0.01,
        "lrf": 0.01,
        "momentum": 0.937,
        "weight_decay": 0.0005,
        "warmup_epochs": (
            0.0
            if args.resume_from is not None
            else 3.0
        ),
        "seed": args.seed,
        "deterministic": True,
        "amp": False,
        "val": True,
        "plots": False,
        "save": True,
        "save_period": args.save_period,
        "project": str(project),
        "name": run_name,
        "exist_ok": False,
        "verbose": True,
    }


def build_trainer(
    baseline: str,
    overrides: dict[str, Any],
    *,
    resume_from: Path | None,
):
    if baseline in {
        "B0",
        "B1",
        "B2",
        "B3",
        "B4",
        "B5",
    }:
        if resume_from is not None:
            raise ValueError(
                "Tensor-only --resume-from is currently "
                "supported only for B6-B8"
            )

        return QVisionDetectionTrainer(
            baseline_id=baseline,
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

    return QuantumDetectionTrainer(
        quantum_baseline_id=baseline,
        quantum_target_layer=10,
        quantum_channels=256,
        quantum_n_qubits=4,
        quantum_n_layers=2,
        quantum_hidden_dim=64,
        quantum_alpha_init=1.0e-3,
        quantum_gamma=0.5,
        quantum_resume_path=resume_from,
        overrides=overrides,
    )


def main() -> None:
    args = parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")

    data_yaml = Path(args.data_yaml).expanduser().resolve()

    if not data_yaml.is_file():
        raise FileNotFoundError(data_yaml)

    resume_from = (
        None
        if args.resume_from is None
        else Path(args.resume_from)
        .expanduser()
        .resolve()
    )

    resume_source_epoch: int | None = None
    resume_source_ema_updates: int | None = None
    resume_source_best_fitness: float | None = None

    if resume_from is not None:
        if args.baseline not in {
            "B6",
            "B7",
            "B8",
        }:
            raise ValueError(
                "--resume-from is supported only for B6-B8"
            )

        if not resume_from.is_file():
            raise FileNotFoundError(resume_from)

        resume_payload = load_quantum_checkpoint(
            resume_from,
            map_location="cpu",
        )

        checkpoint_baseline = resume_payload[
            "model_metadata"
        ]["gate"]["baseline_id"]

        if checkpoint_baseline != args.baseline:
            raise ValueError(
                "Resume baseline mismatch: "
                f"CLI={args.baseline}, "
                f"checkpoint={checkpoint_baseline}"
            )

        resume_source_epoch = int(
            resume_payload["epoch"]
        )
        resume_source_ema_updates = int(
            resume_payload["ema_updates"]
        )
        resume_source_best_fitness = (
            None
            if resume_payload["best_fitness"] is None
            else float(
                resume_payload["best_fitness"]
            )
        )

        completed_epochs = (
            resume_source_epoch + 1
        )

        if args.epochs <= completed_epochs:
            raise ValueError(
                "--epochs is the total target epoch count. "
                f"Checkpoint already completed "
                f"{completed_epochs} epochs, but "
                f"--epochs={args.epochs}."
            )

    server = os.environ.get(
        "QVF_SERVER",
        "UNKNOWN",
    )

    run_name = (
        f"voc_{args.baseline.lower()}_"
        f"seed{args.seed}_{args.run_suffix}"
    )

    overrides = build_overrides(
        args,
        server=server,
        run_name=run_name,
        data_yaml=data_yaml,
    )

    print("===== Q-VISIONFRAME VOC TRAINING =====")
    print("Server    :", server)
    print("Baseline  :", args.baseline)
    print("Run name  :", run_name)
    print("Data YAML :", data_yaml)
    print("CUDA      :", torch.cuda.is_available())
    print("GPU       :", torch.cuda.get_device_name(0))
    print("Resume    :", resume_from)

    if resume_from is not None:
        print(
            "Source epoch      :",
            resume_source_epoch,
        )
        print(
            "Source EMA updates:",
            resume_source_ema_updates,
        )
        print(
            "Source best fitness:",
            resume_source_best_fitness,
        )
        print(
            "Resume start epoch:",
            resume_source_epoch + 1,
        )

    print()
    print(
        json.dumps(
            overrides,
            indent=2,
            sort_keys=True,
        )
    )

    if args.dry_run:
        print()
        print("VOC FULL TRAINING DRY RUN: PASSED")
        return

    start = time.perf_counter()

    trainer = build_trainer(
        args.baseline,
        overrides,
        resume_from=resume_from,
    )

    trainer.train()

    elapsed = time.perf_counter() - start

    report = {
        "server": server,
        "baseline": args.baseline,
        "seed": args.seed,
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "nbs": args.nbs,
        "cache": args.cache,
        "workers": args.workers,
        "device": args.device,
        "data_yaml": str(data_yaml),
        "resume_from": (
            None
            if resume_from is None
            else str(resume_from)
        ),
        "resume_source_epoch": (
            resume_source_epoch
        ),
        "resume_source_ema_updates": (
            resume_source_ema_updates
        ),
        "resume_source_best_fitness": (
            resume_source_best_fitness
        ),
        "elapsed_seconds": elapsed,
        "elapsed_hours": elapsed / 3600.0,
        "save_dir": str(trainer.save_dir),
    }

    report_path = (
        Path(trainer.save_dir)
        / "full_training_runtime.json"
    )

    report_path.write_text(
        json.dumps(
            report,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    print()
    print("Elapsed seconds:", elapsed)
    print("Runtime report :", report_path)
    print()
    print(
        f"VOC {args.baseline} TRAINING: PASSED"
    )


if __name__ == "__main__":
    main()
