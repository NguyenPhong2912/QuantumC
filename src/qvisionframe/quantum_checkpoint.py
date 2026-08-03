from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
from torch import nn

from src.qvisionframe.quantum_detection_model import (
    QuantumGatedDetectionModel,
)
from src.qvisionframe.scientific_state_hash import (
    attach_scientific_state_manifest,
    verify_scientific_state_manifest,
)


QVF_QUANTUM_FORMAT = "qvisionframe.quantum.v1"


def clone_tensor_tree_to_cpu(
    value: Any,
) -> Any:
    """
    Recursively copy tensor-containing structures to CPU.

    The returned tree contains only tensors and ordinary Python
    containers. No nn.Module, optimizer, QNode or simulator runtime
    object is retained.
    """
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()

    if isinstance(value, Mapping):
        return {
            key: clone_tensor_tree_to_cpu(item)
            for key, item in value.items()
        }

    if isinstance(value, tuple):
        return tuple(
            clone_tensor_tree_to_cpu(item)
            for item in value
        )

    if isinstance(value, list):
        return [
            clone_tensor_tree_to_cpu(item)
            for item in value
        ]

    return value


def model_state_to_cpu(
    model: nn.Module,
) -> dict[str, torch.Tensor]:
    return {
        key: tensor.detach().cpu().clone()
        for key, tensor in model.state_dict().items()
    }


def quantum_model_metadata(
    model: QuantumGatedDetectionModel,
) -> dict[str, Any]:
    """
    Return sufficient metadata to reconstruct the model.

    detector_cfg is stored as the YAML dictionary already parsed by
    Ultralytics, avoiding dependence on a model YAML search path when
    loading the checkpoint.
    """
    return {
        "model_class": "QuantumGatedDetectionModel",
        "detector_cfg": clone_tensor_tree_to_cpu(
            model.yaml
        ),
        "nc": int(model.model[-1].nc),
        "gate": model.quantum_checkpoint_metadata(),
    }


def build_quantum_checkpoint(
    *,
    model: QuantumGatedDetectionModel,
    ema_model: QuantumGatedDetectionModel,
    ema_updates: int,
    optimizer: torch.optim.Optimizer | None,
    scaler_state_dict: dict[str, Any] | None,
    epoch: int,
    best_fitness: float | None,
    train_args: dict[str, Any],
    train_metrics: dict[str, Any] | None = None,
    train_results: dict[str, Any] | list[Any] | None = None,
    extra_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if not isinstance(
        model,
        QuantumGatedDetectionModel,
    ):
        raise TypeError(
            "model must be QuantumGatedDetectionModel"
        )

    if not isinstance(
        ema_model,
        QuantumGatedDetectionModel,
    ):
        raise TypeError(
            "ema_model must be QuantumGatedDetectionModel"
        )

    model_metadata = quantum_model_metadata(model)
    ema_metadata = quantum_model_metadata(ema_model)

    if model_metadata != ema_metadata:
        raise RuntimeError(
            "Raw model and EMA reconstruction metadata differ"
        )

    payload = {
        "qvf_format": QVF_QUANTUM_FORMAT,
        "format_version": 1,
        "epoch": int(epoch),
        "best_fitness": (
            None
            if best_fitness is None
            else float(best_fitness)
        ),
        "model_metadata": model_metadata,
        "model_state_dict": model_state_to_cpu(model),
        "ema_state_dict": model_state_to_cpu(ema_model),
        "ema_updates": int(ema_updates),
        "optimizer_state_dict": (
            None
            if optimizer is None
            else clone_tensor_tree_to_cpu(
                optimizer.state_dict()
            )
        ),
        "scaler_state_dict": (
            None
            if scaler_state_dict is None
            else clone_tensor_tree_to_cpu(
                scaler_state_dict
            )
        ),
        "train_args": dict(train_args),
        "train_metrics": (
            {}
            if train_metrics is None
            else dict(train_metrics)
        ),
        "train_results": train_results,
        "extra_metadata": (
            {}
            if extra_metadata is None
            else dict(extra_metadata)
        ),
    }

    attach_scientific_state_manifest(
        payload
    )

    assert_no_runtime_objects(payload)

    # Verify immediately before handing the payload to the caller.
    verify_scientific_state_manifest(
        payload,
        required=True,
    )

    return payload


def assert_no_runtime_objects(
    value: Any,
    *,
    path: str = "checkpoint",
) -> None:
    """
    Reject objects that would reintroduce whole-model pickling.
    """
    if isinstance(value, nn.Module):
        raise TypeError(
            f"{path} contains nn.Module "
            f"{type(value).__name__}"
        )

    if isinstance(value, torch.optim.Optimizer):
        raise TypeError(
            f"{path} contains optimizer object"
        )

    module_name = type(value).__module__

    if module_name.startswith("pennylane"):
        raise TypeError(
            f"{path} contains PennyLane runtime object "
            f"{type(value).__name__}"
        )

    if "pennylane_lightning" in module_name:
        raise TypeError(
            f"{path} contains Lightning runtime object "
            f"{type(value).__name__}"
        )

    if isinstance(value, Mapping):
        for key, item in value.items():
            assert_no_runtime_objects(
                item,
                path=f"{path}.{key}",
            )
        return

    if isinstance(value, (tuple, list)):
        for index, item in enumerate(value):
            assert_no_runtime_objects(
                item,
                path=f"{path}[{index}]",
            )


def save_quantum_checkpoint(
    payload: dict[str, Any],
    path: str | Path,
) -> Path:
    assert_no_runtime_objects(payload)

    output_path = Path(path)
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary_path = output_path.with_suffix(
        output_path.suffix + ".tmp"
    )

    torch.save(
        payload,
        temporary_path,
    )

    temporary_path.replace(output_path)

    return output_path


def load_quantum_checkpoint(
    path: str | Path,
    *,
    map_location: str | torch.device = "cpu",
) -> dict[str, Any]:
    payload = torch.load(
        Path(path),
        map_location=map_location,
        weights_only=True,
    )

    if not isinstance(payload, dict):
        raise TypeError(
            "Quantum checkpoint must be a dictionary"
        )

    if payload.get("qvf_format") != QVF_QUANTUM_FORMAT:
        raise ValueError(
            "Not a Q-VisionFrame quantum checkpoint"
        )

    if payload.get("format_version") != 1:
        raise ValueError(
            "Unsupported quantum checkpoint version: "
            f"{payload.get('format_version')!r}"
        )

    required_keys = {
        "model_metadata",
        "model_state_dict",
        "ema_state_dict",
        "ema_updates",
        "epoch",
    }

    missing = sorted(
        required_keys.difference(payload)
    )

    if missing:
        raise KeyError(
            f"Quantum checkpoint missing keys: {missing}"
        )

    assert_no_runtime_objects(payload)

    # Legacy qvisionframe.quantum.v1 files created before the
    # scientific manifest was introduced remain readable. New files
    # carrying the manifest are always verified.
    verify_scientific_state_manifest(
        payload,
        required=False,
    )

    return payload


def reconstruct_quantum_model(
    payload: dict[str, Any],
    *,
    state_key: str = "model_state_dict",
    device: str | torch.device = "cpu",
    strict: bool = True,
    verbose: bool = False,
) -> QuantumGatedDetectionModel:
    metadata = payload["model_metadata"]
    gate = metadata["gate"]

    model = QuantumGatedDetectionModel(
        cfg=metadata["detector_cfg"],
        nc=int(metadata["nc"]),
        baseline_id=gate["baseline_id"],
        target_layer=int(gate["target_layer"]),
        channels=int(gate["channels"]),
        n_qubits=int(gate["n_qubits"]),
        n_layers=int(gate["n_layers"]),
        hidden_dim=gate["hidden_dim"],
        alpha_init=float(gate["alpha_init"]),
        gamma=float(gate["gamma"]),
        verbose=verbose,
    )

    load_result = model.load_state_dict(
        payload[state_key],
        strict=strict,
    )

    if strict:
        if load_result.missing_keys:
            raise RuntimeError(
                "Missing model state keys: "
                f"{load_result.missing_keys}"
            )

        if load_result.unexpected_keys:
            raise RuntimeError(
                "Unexpected model state keys: "
                f"{load_result.unexpected_keys}"
            )

    model.to(device)

    # The constructor already creates quantum_weights in FP64,
    # but enforce the model invariant after device conversion.
    model._restore_quantum_fp64()

    return model
