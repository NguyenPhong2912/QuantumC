from __future__ import annotations

import json
import traceback
from pathlib import Path
from typing import Any, Callable

import torch

from src.qvisionframe.pqfg import PQFG, PQFGConfig


BASELINES: dict[str, dict[str, Any]] = {
    "B6": {
        "entanglement": "ring",
        "trainable_quantum": False,
    },
    "B7": {
        "entanglement": "none",
        "trainable_quantum": True,
    },
    "B8": {
        "entanglement": "ring",
        "trainable_quantum": True,
    },
}


def make_gate(baseline_id: str) -> PQFG:
    values = BASELINES[baseline_id]

    return PQFG(
        PQFGConfig(
            channels=256,
            n_qubits=4,
            n_layers=2,
            hidden_dim=64,
            alpha_init=1.0e-3,
            entanglement=values["entanglement"],
            gamma=0.5,
            trainable_quantum=values[
                "trainable_quantum"
            ],
        )
    )


def tensor_inventory(
    gate: PQFG,
) -> dict[str, dict[str, Any]]:
    return {
        name: {
            "shape": list(parameter.shape),
            "dtype": str(parameter.dtype),
            "device": str(parameter.device),
            "requires_grad": parameter.requires_grad,
        }
        for name, parameter
        in gate.named_parameters()
    }


def run_attempt(
    name: str,
    function: Callable[[], Any],
) -> dict[str, Any]:
    try:
        result = function()

        print(f"{name:<32}: PASSED")
        print("  result:", result)

        return {
            "passed": True,
            "result": result,
            "error_type": None,
            "error": None,
            "traceback": None,
        }

    except Exception as error:
        print(f"{name:<32}: FAILED")
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


def direct_cuda_forward(
    baseline_id: str,
) -> dict[str, Any]:
    gate = make_gate(baseline_id)

    before = tensor_inventory(gate)

    # Không dùng deepcopy: kiểm tra trực tiếp hành vi
    # nn.Module.cuda() trên PQFG mới tạo.
    gate.cuda()
    gate.eval()

    after = tensor_inventory(gate)

    input_tensor = torch.randn(
        1,
        256,
        4,
        4,
        dtype=torch.float32,
        device="cuda:0",
    )

    with torch.no_grad():
        output = gate(input_tensor)

    return {
        "before": before,
        "after": after,
        "output_shape": list(output.shape),
        "output_dtype": str(output.dtype),
        "output_device": str(output.device),
    }


def direct_cuda_backward(
    baseline_id: str,
) -> dict[str, Any]:
    gate = make_gate(baseline_id)
    gate.cuda()
    gate.train()
    gate.zero_grad(set_to_none=True)

    input_tensor = torch.randn(
        1,
        256,
        4,
        4,
        dtype=torch.float32,
        device="cuda:0",
    )

    output = gate(input_tensor)
    loss = output.square().mean()
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
            "dtype": (
                str(parameter.grad.dtype)
                if parameter.grad is not None
                else None
            ),
            "device": (
                str(parameter.grad.device)
                if parameter.grad is not None
                else None
            ),
            "norm": (
                float(
                    parameter.grad.detach()
                    .float()
                    .norm()
                    .cpu()
                )
                if parameter.grad is not None
                else None
            ),
        }
        for name, parameter
        in gate.named_parameters()
    }

    return {
        "loss": float(loss.detach().cpu()),
        "gradients": gradients,
    }


def direct_half_forward(
    baseline_id: str,
) -> dict[str, Any]:
    gate = make_gate(baseline_id)

    before = tensor_inventory(gate)

    # Không dùng deepcopy.
    gate.half()
    gate.eval()

    after = tensor_inventory(gate)

    input_tensor = torch.randn(
        1,
        256,
        4,
        4,
        dtype=torch.float16,
    )

    with torch.no_grad():
        output = gate(input_tensor)

    return {
        "before": before,
        "after": after,
        "output_shape": list(output.shape),
        "output_dtype": str(output.dtype),
        "output_device": str(output.device),
    }


def direct_float_after_half(
    baseline_id: str,
) -> dict[str, Any]:
    gate = make_gate(baseline_id)

    gate.half()
    gate.float()
    gate.eval()

    inventory = tensor_inventory(gate)

    input_tensor = torch.randn(
        1,
        256,
        4,
        4,
        dtype=torch.float32,
    )

    with torch.no_grad():
        output = gate(input_tensor)

    return {
        "parameters": inventory,
        "output_shape": list(output.shape),
        "output_dtype": str(output.dtype),
        "output_device": str(output.device),
    }


def main() -> None:
    print("===== PQFG DIRECT CONVERSION AUDIT =====")

    root = Path(__file__).resolve().parents[1]

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")

    report: dict[str, Any] = {}

    for baseline_id in BASELINES:
        print()
        print(f"===== {baseline_id} =====")

        results = {
            "direct_cuda_forward": run_attempt(
                "direct_cuda_forward",
                lambda baseline_id=baseline_id:
                    direct_cuda_forward(baseline_id),
            ),
            "direct_cuda_backward": run_attempt(
                "direct_cuda_backward",
                lambda baseline_id=baseline_id:
                    direct_cuda_backward(baseline_id),
            ),
            "direct_half_forward": run_attempt(
                "direct_half_forward",
                lambda baseline_id=baseline_id:
                    direct_half_forward(baseline_id),
            ),
            "direct_float_after_half": run_attempt(
                "direct_float_after_half",
                lambda baseline_id=baseline_id:
                    direct_float_after_half(
                        baseline_id
                    ),
            ),
        }

        report[baseline_id] = results

    output_path = (
        root
        / "outputs/inventory/"
        "pqfg_direct_conversion_audit.json"
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
    print("Report:", output_path)
    print("PQFG DIRECT CONVERSION AUDIT: COMPLETED")


if __name__ == "__main__":
    main()
