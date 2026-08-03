from __future__ import annotations

from typing import Any

import torch
from ultralytics import YOLO


def collect_tensors(obj: Any) -> list[torch.Tensor]:
    """Recursively collect differentiable tensors from nested YOLO outputs."""
    tensors: list[torch.Tensor] = []

    if torch.is_tensor(obj):
        tensors.append(obj)

    elif isinstance(obj, dict):
        for value in obj.values():
            tensors.extend(collect_tensors(value))

    elif isinstance(obj, (list, tuple)):
        for value in obj:
            tensors.extend(collect_tensors(value))

    return tensors


def main() -> None:
    print("===== YOLO BACKWARD CHECK =====")

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")

    device = torch.device("cuda:0")

    model = YOLO("yolo11n.yaml")
    network = model.model.to(device).train()

    x = torch.randn(
        1,
        3,
        320,
        320,
        device=device,
        requires_grad=True,
    )

    network.zero_grad(set_to_none=True)

    outputs = network(x)
    tensors = collect_tensors(outputs)

    differentiable_tensors = [
        tensor
        for tensor in tensors
        if tensor.is_floating_point() and tensor.requires_grad
    ]

    if not differentiable_tensors:
        raise RuntimeError(
            "No differentiable tensor was found in YOLO training output"
        )

    # Surrogate objective only for validating the autograd graph.
    loss = sum(
        tensor.float().square().mean()
        for tensor in differentiable_tensors
    )

    loss.backward()

    parameters_with_grad = [
        parameter
        for parameter in network.parameters()
        if parameter.requires_grad and parameter.grad is not None
    ]

    finite_parameter_grads = all(
        torch.isfinite(parameter.grad).all().item()
        for parameter in parameters_with_grad
    )

    parameter_grad_norm = torch.sqrt(
        sum(
            parameter.grad.detach().float().square().sum()
            for parameter in parameters_with_grad
        )
    )

    print("GPU                 :", torch.cuda.get_device_name(0))
    print("Output type         :", type(outputs).__name__)
    print("Tensor count        :", len(tensors))
    print("Differentiable count:", len(differentiable_tensors))
    print(
        "Tensor shapes       :",
        [tuple(tensor.shape) for tensor in differentiable_tensors],
    )
    print("Surrogate loss      :", float(loss.detach()))
    print("Input grad exists   :", x.grad is not None)
    print(
        "Input grad finite   :",
        bool(torch.isfinite(x.grad).all()) if x.grad is not None else False,
    )
    print(
        "Input grad norm     :",
        float(x.grad.norm()) if x.grad is not None else 0.0,
    )
    print("Params with grad    :", len(parameters_with_grad))
    print("Param grads finite  :", finite_parameter_grads)
    print("Total grad norm     :", float(parameter_grad_norm))

    assert x.grad is not None
    assert torch.isfinite(x.grad).all()
    assert float(x.grad.norm()) > 0.0

    assert parameters_with_grad
    assert finite_parameter_grads
    assert float(parameter_grad_norm) > 0.0

    print("YOLO BACKWARD TEST  : PASSED")


if __name__ == "__main__":
    main()
