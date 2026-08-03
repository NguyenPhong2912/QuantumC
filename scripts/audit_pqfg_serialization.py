from __future__ import annotations

import copy
import importlib
import inspect
import io
import json
import traceback
from dataclasses import fields, is_dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn


def describe_tensor(
    tensor: torch.Tensor,
) -> dict[str, Any]:
    return {
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
        "device": str(tensor.device),
        "requires_grad": bool(
            getattr(tensor, "requires_grad", False)
        ),
    }


def attempt(
    name: str,
    function,
) -> dict[str, Any]:
    try:
        value = function()

        return {
            "name": name,
            "passed": True,
            "result": value,
            "error_type": None,
            "error": None,
            "traceback": None,
        }

    except Exception as error:
        return {
            "name": name,
            "passed": False,
            "result": None,
            "error_type": type(error).__name__,
            "error": str(error),
            "traceback": traceback.format_exc(),
        }


def main() -> None:
    print("===== PQFG API AND SERIALIZATION AUDIT =====")

    root = Path(__file__).resolve().parents[1]

    module = importlib.import_module(
        "src.qvisionframe.pqfg"
    )

    public_classes: list[dict[str, Any]] = []

    print()
    print("===== PUBLIC CLASSES =====")

    for name in sorted(vars(module)):
        if name.startswith("_"):
            continue

        value = getattr(module, name)

        if not inspect.isclass(value):
            continue

        if value.__module__ != module.__name__:
            continue

        try:
            signature = str(
                inspect.signature(value)
            )
        except (TypeError, ValueError):
            signature = "<unavailable>"

        record = {
            "name": name,
            "qualified_name": (
                f"{module.__name__}.{name}"
            ),
            "signature": signature,
            "is_nn_module": (
                issubclass(value, nn.Module)
            ),
            "is_dataclass": is_dataclass(value),
            "fields": [],
        }

        if is_dataclass(value):
            for field in fields(value):
                record["fields"].append(
                    {
                        "name": field.name,
                        "type": str(field.type),
                        "default": repr(
                            field.default
                        ),
                    }
                )

        public_classes.append(record)

        print(
            f"{name:<40} "
            f"nn.Module={record['is_nn_module']!s:<5} "
            f"dataclass={record['is_dataclass']!s:<5}"
        )
        print(
            "  signature:",
            signature,
        )

    config_class = getattr(
        module,
        "PQFGConfig",
        None,
    )

    if config_class is None:
        raise RuntimeError(
            "PQFGConfig was not found"
        )

    gate_candidates = [
        getattr(module, record["name"])
        for record in public_classes
        if record["is_nn_module"]
        and "pqfg" in record["name"].lower()
    ]

    if not gate_candidates:
        gate_candidates = [
            getattr(module, record["name"])
            for record in public_classes
            if record["is_nn_module"]
            and "quantum" in record["name"].lower()
        ]

    if len(gate_candidates) != 1:
        raise RuntimeError(
            "Expected exactly one PQFG nn.Module class; "
            f"found {[item.__name__ for item in gate_candidates]}"
        )

    gate_class = gate_candidates[0]

    print()
    print("Selected gate class:", gate_class.__name__)

    configurations = {
        "B6": config_class(
            channels=256,
            n_qubits=4,
            n_layers=2,
            hidden_dim=64,
            alpha_init=1.0e-3,
            entanglement="ring",
            gamma=0.5,
            trainable_quantum=False,
        ),
        "B7": config_class(
            channels=256,
            n_qubits=4,
            n_layers=2,
            hidden_dim=64,
            alpha_init=1.0e-3,
            entanglement="none",
            gamma=0.5,
            trainable_quantum=True,
        ),
        "B8": config_class(
            channels=256,
            n_qubits=4,
            n_layers=2,
            hidden_dim=64,
            alpha_init=1.0e-3,
            entanglement="ring",
            gamma=0.5,
            trainable_quantum=True,
        ),
    }

    report: dict[str, Any] = {
        "module": module.__name__,
        "classes": public_classes,
        "gate_class": gate_class.__name__,
        "baselines": {},
    }

    for baseline_id, config in configurations.items():
        print()
        print(
            f"===== {baseline_id} "
            f"entanglement={config.entanglement} "
            f"trainable_quantum="
            f"{config.trainable_quantum} ====="
        )

        torch.manual_seed(42)

        gate = gate_class(config)

        parameters = {
            name: describe_tensor(parameter)
            for name, parameter
            in gate.named_parameters()
        }

        buffers = {
            name: describe_tensor(buffer)
            for name, buffer
            in gate.named_buffers()
        }

        state = {
            name: describe_tensor(tensor)
            for name, tensor
            in gate.state_dict().items()
        }

        total_parameters = sum(
            parameter.numel()
            for parameter in gate.parameters()
        )

        trainable_parameters = sum(
            parameter.numel()
            for parameter in gate.parameters()
            if parameter.requires_grad
        )

        print(
            "Total parameters    :",
            total_parameters,
        )
        print(
            "Trainable parameters:",
            trainable_parameters,
        )

        print("Named parameters:")

        for name, description in parameters.items():
            print(
                f"  {name:<36} "
                f"{description['shape']} "
                f"{description['dtype']} "
                f"{description['device']} "
                f"grad={description['requires_grad']}"
            )

        print("Named buffers:")

        if not buffers:
            print("  <none>")

        for name, description in buffers.items():
            print(
                f"  {name:<36} "
                f"{description['shape']} "
                f"{description['dtype']} "
                f"{description['device']}"
            )

        input_cpu = torch.randn(
            2,
            256,
            4,
            4,
            dtype=torch.float32,
        )

        forward_cpu = attempt(
            "forward_cpu_fp32",
            lambda: {
                "shape": list(
                    gate(input_cpu).shape
                ),
                "dtype": str(
                    gate(input_cpu).dtype
                ),
                "device": str(
                    gate(input_cpu).device
                ),
            },
        )

        backward_cpu = attempt(
            "backward_cpu_fp32",
            lambda: backward_check(
                gate,
                input_cpu,
            ),
        )

        deepcopy_result = attempt(
            "deepcopy",
            lambda: describe_module(
                copy.deepcopy(gate)
            ),
        )

        serialization_result = attempt(
            "torch_save_load",
            lambda: save_load_check(gate),
        )

        half_result = attempt(
            "deepcopy_half",
            lambda: half_check(gate),
        )

        cuda_result = attempt(
            "deepcopy_cuda_forward",
            lambda: cuda_check(gate),
        )

        checks = [
            forward_cpu,
            backward_cpu,
            deepcopy_result,
            serialization_result,
            half_result,
            cuda_result,
        ]

        for check in checks:
            print(
                f"{check['name']:<28}: "
                f"{'PASSED' if check['passed'] else 'FAILED'}"
            )

            if check["passed"]:
                print(
                    "  result:",
                    check["result"],
                )
            else:
                print(
                    "  error :",
                    check["error_type"],
                    check["error"],
                )

        report["baselines"][baseline_id] = {
            "config": {
                "channels": config.channels,
                "n_qubits": config.n_qubits,
                "n_layers": config.n_layers,
                "hidden_dim": config.hidden_dim,
                "alpha_init": config.alpha_init,
                "entanglement": config.entanglement,
                "gamma": config.gamma,
                "trainable_quantum": (
                    config.trainable_quantum
                ),
            },
            "total_parameters": total_parameters,
            "trainable_parameters": (
                trainable_parameters
            ),
            "parameters": parameters,
            "buffers": buffers,
            "state": state,
            "checks": checks,
        }

    output_path = (
        root
        / "outputs/inventory/"
        "pqfg_serialization_audit.json"
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path.write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )

    print()
    print("Audit report:", output_path)
    print("PQFG API AND SERIALIZATION AUDIT: COMPLETED")


def describe_module(
    module: nn.Module,
) -> dict[str, Any]:
    return {
        "type": type(module).__name__,
        "parameters": {
            name: describe_tensor(parameter)
            for name, parameter
            in module.named_parameters()
        },
    }


def backward_check(
    gate: nn.Module,
    input_tensor: torch.Tensor,
) -> dict[str, Any]:
    gate.zero_grad(set_to_none=True)

    value = gate(
        input_tensor.clone()
    )

    loss = value.square().mean()
    loss.backward()

    gradients = {
        name: {
            "present": parameter.grad is not None,
            "finite": (
                bool(
                    torch.isfinite(
                        parameter.grad
                    ).all()
                )
                if parameter.grad is not None
                else None
            ),
            "norm": (
                float(
                    parameter.grad.detach()
                    .float()
                    .norm()
                )
                if parameter.grad is not None
                else None
            ),
        }
        for name, parameter
        in gate.named_parameters()
    }

    return {
        "loss": float(loss.detach()),
        "gradients": gradients,
    }


def save_load_check(
    gate: nn.Module,
) -> dict[str, Any]:
    buffer = io.BytesIO()

    torch.save(
        gate,
        buffer,
    )

    byte_size = buffer.tell()
    buffer.seek(0)

    loaded = torch.load(
        buffer,
        map_location="cpu",
        weights_only=False,
    )

    input_tensor = torch.randn(
        1,
        256,
        4,
        4,
        dtype=torch.float32,
    )

    with torch.no_grad():
        output = loaded(input_tensor)

    return {
        "loaded_type": type(loaded).__name__,
        "serialized_bytes": byte_size,
        "output_shape": list(output.shape),
        "output_dtype": str(output.dtype),
        "output_device": str(output.device),
    }


def half_check(
    gate: nn.Module,
) -> dict[str, Any]:
    copied = copy.deepcopy(gate)
    copied.half()

    parameter_dtypes = {
        name: str(parameter.dtype)
        for name, parameter
        in copied.named_parameters()
    }

    input_tensor = torch.randn(
        1,
        256,
        4,
        4,
        dtype=torch.float16,
    )

    with torch.no_grad():
        output = copied(input_tensor)

    return {
        "parameter_dtypes": parameter_dtypes,
        "output_shape": list(output.shape),
        "output_dtype": str(output.dtype),
        "output_device": str(output.device),
    }


def cuda_check(
    gate: nn.Module,
) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is unavailable"
        )

    copied = copy.deepcopy(gate)
    copied.cuda()

    parameter_devices = {
        name: str(parameter.device)
        for name, parameter
        in copied.named_parameters()
    }

    input_tensor = torch.randn(
        1,
        256,
        4,
        4,
        dtype=torch.float32,
        device="cuda:0",
    )

    with torch.no_grad():
        output = copied(input_tensor)

    return {
        "parameter_devices": parameter_devices,
        "output_shape": list(output.shape),
        "output_dtype": str(output.dtype),
        "output_device": str(output.device),
    }


if __name__ == "__main__":
    main()
