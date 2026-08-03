from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from src.qvisionframe.pqfg import PQFGConfig
from src.qvisionframe.yolo_hybrid import HybridPlacementConfig


@dataclass(frozen=True)
class ExperimentMetadata:
    name: str
    seed: int
    description: str = ""


@dataclass(frozen=True)
class DetectorConfig:
    detector: str
    architecture: str
    pretrained: bool = False


@dataclass(frozen=True)
class RuntimeConfig:
    detector_device: str = "cuda:0"
    quantum_device: str = "cpu"
    dtype: str = "float32"
    quantum_dtype: str = "float64"


@dataclass(frozen=True)
class QVisionFrameConfig:
    experiment: ExperimentMetadata
    model: DetectorConfig
    placement: HybridPlacementConfig
    pqfg: PQFGConfig
    runtime: RuntimeConfig


def require_mapping(
    mapping: dict[str, Any],
    key: str,
) -> dict[str, Any]:
    value = mapping.get(key)

    if not isinstance(value, dict):
        raise ValueError(
            f"Configuration section '{key}' must be a mapping"
        )

    return value


def load_config(path: str | Path) -> QVisionFrameConfig:
    config_path = Path(path)

    if not config_path.is_file():
        raise FileNotFoundError(
            f"Configuration file not found: {config_path}"
        )

    with config_path.open("r", encoding="utf-8") as file:
        raw = yaml.safe_load(file)

    if not isinstance(raw, dict):
        raise ValueError("Root YAML object must be a mapping")

    experiment_raw = require_mapping(raw, "experiment")
    model_raw = require_mapping(raw, "model")
    placement_raw = require_mapping(raw, "placement")
    pqfg_raw = require_mapping(raw, "pqfg")
    runtime_raw = require_mapping(raw, "runtime")

    if not bool(pqfg_raw.get("enabled", True)):
        raise ValueError(
            "This configuration loader currently expects PQFG enabled"
        )

    experiment = ExperimentMetadata(
        name=str(experiment_raw["name"]),
        seed=int(experiment_raw["seed"]),
        description=str(
            experiment_raw.get("description", "")
        ).strip(),
    )

    model = DetectorConfig(
        detector=str(model_raw["detector"]),
        architecture=str(model_raw["architecture"]),
        pretrained=bool(model_raw.get("pretrained", False)),
    )

    placement = HybridPlacementConfig(
        target_layer=int(placement_raw["target_layer"]),
        channels=int(placement_raw["channels"]),
    )

    pqfg = PQFGConfig(
        channels=placement.channels,
        n_qubits=int(pqfg_raw["n_qubits"]),
        n_layers=int(pqfg_raw["n_layers"]),
        hidden_dim=(
            None
            if pqfg_raw.get("hidden_dim") is None
            else int(pqfg_raw["hidden_dim"])
        ),
        alpha_init=float(pqfg_raw["alpha_init"]),
        entanglement=str(pqfg_raw["entanglement"]),
        gamma=float(pqfg_raw.get("gamma", 0.5)),
        trainable_quantum=bool(
            pqfg_raw.get("trainable_quantum", True)
        ),
    )

    runtime = RuntimeConfig(
        detector_device=str(
            runtime_raw.get("detector_device", "cuda:0")
        ),
        quantum_device=str(
            runtime_raw.get("quantum_device", "cpu")
        ),
        dtype=str(runtime_raw.get("dtype", "float32")),
        quantum_dtype=str(
            runtime_raw.get("quantum_dtype", "float64")
        ),
    )

    if runtime.quantum_device != "cpu":
        raise ValueError(
            "lightning.qubit currently requires quantum_device=cpu"
        )

    if runtime.dtype != "float32":
        raise ValueError(
            "The current YOLO integration expects dtype=float32"
        )

    if runtime.quantum_dtype != "float64":
        raise ValueError(
            "The current PennyLane integration expects "
            "quantum_dtype=float64"
        )

    return QVisionFrameConfig(
        experiment=experiment,
        model=model,
        placement=placement,
        pqfg=pqfg,
        runtime=runtime,
    )
