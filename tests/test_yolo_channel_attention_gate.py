from __future__ import annotations

import torch
from ultralytics import YOLO
from ultralytics.cfg import get_cfg
from ultralytics.utils import DEFAULT_CFG

from src.qvisionframe.classical_gate import (
    ChannelAttentionGateConfig,
)
from src.qvisionframe.yolo_channel_attention import (
    YOLOWithChannelAttentionGate,
)
from src.qvisionframe.yolo_hybrid import HybridPlacementConfig


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
    print("===== YOLO + CHANNEL ATTENTION GATE CHECK =====")

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

    gate_config = ChannelAttentionGateConfig(
        channels=256,
        reduction=16,
        alpha_init=1.0e-3,
    )

    placement = HybridPlacementConfig(
        target_layer=10,
        channels=256,
    )

    model = YOLOWithChannelAttentionGate(
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

    first_weight_grad = grad_norm(
        model.gate.shared_mlp[0].weight
    )
    first_bias_grad = grad_norm(
        model.gate.shared_mlp[0].bias
    )
    second_weight_grad = grad_norm(
        model.gate.shared_mlp[2].weight
    )
    second_bias_grad = grad_norm(
        model.gate.shared_mlp[2].bias
    )
    alpha_grad = grad_norm(
        model.gate.alpha
    )

    all_gate_grads_finite = all(
        parameter.grad is not None
        and bool(torch.isfinite(parameter.grad).all())
        for parameter in model.gate.parameters()
    )

    expected_parameters = (
        gate_config.channels
        * model.gate.hidden_channels
        + model.gate.hidden_channels
        + model.gate.hidden_channels
        * gate_config.channels
        + gate_config.channels
        + 1
    )

    print("GPU                  :", torch.cuda.get_device_name(0))
    print("Target layer         :", placement.target_layer)
    print("Feature shape        :", model.last_feature_shape)
    print("Hook calls           :", model.hook_calls)
    print("Reduction ratio      :", gate_config.reduction)
    print("Hidden channels      :", model.gate.hidden_channels)
    print("Gate parameters      :", parameter_count(model.gate))
    print("Expected parameters  :", expected_parameters)
    print(
        "Loss components      :",
        loss_components.detach().cpu().tolist(),
    )
    print("Total loss           :", float(total_loss.detach()))
    print(
        "Loss items           :",
        loss_items.detach().cpu().tolist(),
    )
    print("First weight grad    :", first_weight_grad)
    print("First bias grad      :", first_bias_grad)
    print("Second weight grad   :", second_weight_grad)
    print("Second bias grad     :", second_bias_grad)
    print("Alpha grad norm      :", alpha_grad)
    print("All gate grads finite:", all_gate_grads_finite)

    assert model.hook_calls == 1
    assert model.last_feature_shape == (2, 256, 10, 10)
    assert model.gate.hidden_channels == 16

    assert parameter_count(model.gate) == expected_parameters
    assert expected_parameters == 8465

    assert torch.isfinite(loss_components).all()
    assert torch.isfinite(loss_items).all()
    assert float(total_loss.detach()) > 0.0

    assert first_weight_grad > 0.0
    assert first_bias_grad > 0.0
    assert second_weight_grad > 0.0
    assert second_bias_grad > 0.0
    assert alpha_grad > 0.0
    assert all_gate_grads_finite

    model.close()

    print("CHANNEL ATTENTION GATE: PASSED")


if __name__ == "__main__":
    main()
