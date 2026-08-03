from __future__ import annotations

import torch
from ultralytics import YOLO
from ultralytics.cfg import get_cfg
from ultralytics.utils import DEFAULT_CFG

from src.qvisionframe.classical_gate import (
    LinearGateConfig,
)
from src.qvisionframe.yolo_hybrid import HybridPlacementConfig
from src.qvisionframe.yolo_linear import YOLOWithLinearGate


def grad_norm(parameter: torch.Tensor) -> float:
    if parameter.grad is None:
        return 0.0

    return float(
        parameter.grad.detach().float().norm().cpu()
    )


def parameter_count(module: torch.nn.Module) -> int:
    return sum(
        parameter.numel()
        for parameter in module.parameters()
        if parameter.requires_grad
    )


def main() -> None:
    print("===== YOLO + LINEAR GATE CHECK =====")

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")

    torch.manual_seed(42)
    device = torch.device("cuda:0")

    yolo = YOLO("yolo11n.yaml")
    network = yolo.model.to(device).train()

    network.args = get_cfg(
        cfg=DEFAULT_CFG,
        overrides=yolo.overrides,
    )
    network.criterion = None

    gate_config = LinearGateConfig(
        channels=256,
        alpha_init=1.0e-3,
    )

    placement = HybridPlacementConfig(
        target_layer=10,
        channels=256,
    )

    model = YOLOWithLinearGate(
        yolo_network=network,
        gate_config=gate_config,
        placement=placement,
    ).to(device)

    model.zero_grad(set_to_none=True)

    images = torch.rand(
        2,
        3,
        320,
        320,
        dtype=torch.float32,
        device=device,
    )

    classes = torch.tensor(
        [[0.0], [5.0], [2.0], [7.0]],
        dtype=torch.float32,
        device=device,
    )

    boxes = torch.tensor(
        [
            [0.30, 0.35, 0.20, 0.25],
            [0.70, 0.65, 0.15, 0.20],
            [0.40, 0.45, 0.25, 0.30],
            [0.75, 0.30, 0.18, 0.22],
        ],
        dtype=torch.float32,
        device=device,
    )

    batch_indices = torch.tensor(
        [0, 0, 1, 1],
        dtype=torch.long,
        device=device,
    )

    batch = {
        "img": images,
        "cls": classes,
        "bboxes": boxes,
        "batch_idx": batch_indices,
    }

    loss_components, loss_items = model(batch)
    total_loss = loss_components.sum()

    total_loss.backward()

    linear_weight_grad = grad_norm(
        model.gate.linear.weight
    )
    linear_bias_grad = grad_norm(
        model.gate.linear.bias
    )
    alpha_grad = grad_norm(
        model.gate.alpha
    )

    all_gate_grads_finite = all(
        parameter.grad is not None
        and bool(torch.isfinite(parameter.grad).all())
        for parameter in model.gate.parameters()
    )

    print("GPU                  :", torch.cuda.get_device_name(0))
    print("Target layer         :", placement.target_layer)
    print("Feature shape        :", model.last_feature_shape)
    print("Hook calls           :", model.hook_calls)
    print("Gate parameters      :", parameter_count(model.gate))
    print(
        "Loss components      :",
        loss_components.detach().cpu().tolist(),
    )
    print("Total loss           :", float(total_loss.detach()))
    print(
        "Loss items           :",
        loss_items.detach().cpu().tolist(),
    )
    print("Linear weight grad   :", linear_weight_grad)
    print("Linear bias grad     :", linear_bias_grad)
    print("Alpha grad norm      :", alpha_grad)
    print("All gate grads finite:", all_gate_grads_finite)

    expected_parameters = (
        (2 * gate_config.channels) * gate_config.channels
        + gate_config.channels
        + 1
    )

    print("Expected parameters  :", expected_parameters)

    assert model.hook_calls == 1
    assert model.last_feature_shape == (2, 256, 10, 10)
    assert parameter_count(model.gate) == expected_parameters
    assert expected_parameters == 131329

    assert torch.isfinite(loss_components).all()
    assert torch.isfinite(loss_items).all()
    assert float(total_loss.detach()) > 0.0

    assert linear_weight_grad > 0.0
    assert linear_bias_grad > 0.0
    assert alpha_grad > 0.0
    assert all_gate_grads_finite

    model.close()

    print("LINEAR GATE          : PASSED")


if __name__ == "__main__":
    main()
