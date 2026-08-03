from __future__ import annotations

import copy
import inspect
import io
import json
import traceback
from pathlib import Path
from typing import Any, Callable

import torch
from ultralytics.utils.torch_utils import ModelEMA

from src.qvisionframe.quantum_detection_model import (
    QuantumGatedDetectionModel,
)


def attempt(
    name: str,
    function: Callable[[], Any],
) -> dict[str, Any]:
    try:
        result = function()

        print(f"{name:<38}: PASSED")
        print("  result:", result)

        return {
            "passed": True,
            "result": result,
            "error_type": None,
            "error": None,
            "traceback": None,
        }

    except Exception as error:
        print(f"{name:<38}: FAILED")
        print(
            "  error :",
            type(error).__name__,
            str(error),
        )

        return {
            "passed": False,
            "result": None,
            "error_type": type(error).__name__,
            "error": str(error),
            "traceback": traceback.format_exc(),
        }


def make_model() -> QuantumGatedDetectionModel:
    torch.manual_seed(42)

    return QuantumGatedDetectionModel(
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


def parameter_inventory(
    model: torch.nn.Module,
) -> dict[str, dict[str, Any]]:
    return {
        name: {
            "shape": list(parameter.shape),
            "dtype": str(parameter.dtype),
            "device": str(parameter.device),
            "requires_grad": (
                parameter.requires_grad
            ),
        }
        for name, parameter
        in model.named_parameters()
        if name.startswith("gate.")
    }


def object_attribute_inventory(
    value: Any,
) -> dict[str, str]:
    output: dict[str, str] = {}

    for name, attribute in vars(value).items():
        if name.startswith("_"):
            continue

        output[name] = (
            f"{type(attribute).__module__}."
            f"{type(attribute).__qualname__}"
        )

    return output


def deepcopy_gate_check() -> dict[str, Any]:
    model = make_model()
    copied = copy.deepcopy(model.gate)

    return {
        "type": type(copied).__name__,
        "state_keys": list(
            copied.state_dict().keys()
        ),
    }


def deepcopy_model_check() -> dict[str, Any]:
    model = make_model()
    copied = copy.deepcopy(model)

    return {
        "type": type(copied).__name__,
        "baseline_id": copied.baseline_id,
        "gate_type": type(copied.gate).__name__,
        "state_tensors": len(
            copied.state_dict()
        ),
        "hook_handle_type": (
            type(
                copied._gate_hook_handle
            ).__name__
        ),
    }


def model_ema_check() -> dict[str, Any]:
    model = make_model().cuda()

    ema = ModelEMA(model)

    ema_model = ema.ema

    return {
        "ema_type": type(ema_model).__name__,
        "baseline_id": ema_model.baseline_id,
        "gate_type": type(
            ema_model.gate
        ).__name__,
        "quantum_dtype": str(
            ema_model.gate.quantum_weights.dtype
        ),
        "quantum_device": str(
            ema_model.gate.quantum_weights.device
        ),
        "state_tensors": len(
            ema_model.state_dict()
        ),
    }


def tensor_only_checkpoint_check() -> dict[str, Any]:
    model = make_model().cuda()

    state = {
        key: tensor.detach().cpu().clone()
        for key, tensor
        in model.state_dict().items()
    }

    payload = {
        "format_version": 1,
        "model_class": (
            "QuantumGatedDetectionModel"
        ),
        "metadata": (
            model.quantum_checkpoint_metadata()
        ),
        "model_state_dict": state,
    }

    memory = io.BytesIO()

    torch.save(
        payload,
        memory,
    )

    byte_size = memory.tell()
    memory.seek(0)

    loaded = torch.load(
        memory,
        map_location="cpu",
        weights_only=True,
    )

    restored = QuantumGatedDetectionModel(
        cfg="yolo11n.yaml",
        nc=20,
        baseline_id=loaded[
            "metadata"
        ]["baseline_id"],
        target_layer=loaded[
            "metadata"
        ]["target_layer"],
        channels=loaded[
            "metadata"
        ]["channels"],
        n_qubits=loaded[
            "metadata"
        ]["n_qubits"],
        n_layers=loaded[
            "metadata"
        ]["n_layers"],
        hidden_dim=loaded[
            "metadata"
        ]["hidden_dim"],
        alpha_init=loaded[
            "metadata"
        ]["alpha_init"],
        gamma=loaded[
            "metadata"
        ]["gamma"],
        verbose=False,
    )

    result = restored.load_state_dict(
        loaded["model_state_dict"],
        strict=True,
    )

    exact = 0
    total = 0
    max_difference = 0.0

    restored_state = restored.state_dict()

    for key, source_tensor in state.items():
        target_tensor = restored_state[key]

        total += 1

        if torch.equal(
            source_tensor,
            target_tensor,
        ):
            exact += 1

        difference = (
            source_tensor.float()
            - target_tensor.float()
        )

        if difference.numel():
            max_difference = max(
                max_difference,
                float(
                    difference.abs().max()
                ),
            )

    return {
        "serialized_bytes": byte_size,
        "missing_keys": (
            result.missing_keys
        ),
        "unexpected_keys": (
            result.unexpected_keys
        ),
        "exact_state_tensors": exact,
        "total_state_tensors": total,
        "max_state_difference": (
            max_difference
        ),
        "restored_quantum_dtype": str(
            restored.gate.quantum_weights.dtype
        ),
    }


def main() -> None:
    print(
        "===== QUANTUM MODEL EMA AUDIT ====="
    )

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is unavailable"
        )

    root = Path(__file__).resolve().parents[1]

    model = make_model()
    gate = model.gate

    print()
    print("Model class :", type(model).__name__)
    print("Gate class  :", type(gate).__name__)

    print()
    print("===== CONSTRUCTOR SIGNATURES =====")
    print(
        "Quantum model:",
        inspect.signature(
            QuantumGatedDetectionModel
        ),
    )
    print(
        "PQFG gate vars:",
        object_attribute_inventory(gate),
    )

    print()
    print("===== GATE PARAMETER INVENTORY =====")

    for name, values in parameter_inventory(
        model
    ).items():
        print(
            f"{name:<32} "
            f"{values['shape']} "
            f"{values['dtype']} "
            f"{values['device']} "
            f"grad={values['requires_grad']}"
        )

    model.close_gate_hook()
    del model
    del gate

    print()
    print("===== COPY / EMA / STATE CHECKS =====")

    results = {
        "deepcopy_gate": attempt(
            "copy.deepcopy(PQFG)",
            deepcopy_gate_check,
        ),
        "deepcopy_model": attempt(
            "copy.deepcopy(quantum model)",
            deepcopy_model_check,
        ),
        "model_ema": attempt(
            "ModelEMA(quantum model)",
            model_ema_check,
        ),
        "tensor_only_checkpoint": attempt(
            "tensor-only checkpoint",
            tensor_only_checkpoint_check,
        ),
    }

    output_path = (
        root
        / "outputs/inventory/"
        "quantum_model_ema_audit.json"
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path.write_text(
        json.dumps(results, indent=2) + "\n",
        encoding="utf-8",
    )

    print()
    print("Report:", output_path)
    print(
        "QUANTUM MODEL EMA AUDIT: COMPLETED"
    )


if __name__ == "__main__":
    main()
