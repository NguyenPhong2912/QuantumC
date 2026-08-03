from __future__ import annotations

import torch
from ultralytics.cfg import get_cfg
from ultralytics.utils import DEFAULT_CFG

from src.qvisionframe.gated_detection_model import (
    GatedDetectionModel,
)


def parameter_count(module: torch.nn.Module) -> int:
    return sum(
        parameter.numel()
        for parameter in module.parameters()
    )


def grad_norm(parameter: torch.Tensor) -> float:
    if parameter.grad is None:
        return 0.0

    return float(
        parameter.grad.detach().float().norm().cpu()
    )


def make_batch(device: torch.device) -> dict[str, torch.Tensor]:
    return {
        "img": torch.rand(
            2,
            3,
            320,
            320,
            device=device,
        ),
        "cls": torch.tensor(
            [[0.0], [5.0], [2.0], [7.0]],
            dtype=torch.float32,
            device=device,
        ),
        "bboxes": torch.tensor(
            [
                [0.30, 0.35, 0.20, 0.25],
                [0.70, 0.65, 0.15, 0.20],
                [0.40, 0.45, 0.25, 0.30],
                [0.75, 0.30, 0.18, 0.22],
            ],
            dtype=torch.float32,
            device=device,
        ),
        "batch_idx": torch.tensor(
            [0, 0, 1, 1],
            dtype=torch.long,
            device=device,
        ),
    }


def main() -> None:
    print("===== TRAINER-COMPATIBLE B1 MODEL CHECK =====")

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")

    torch.manual_seed(42)
    device = torch.device("cuda:0")

    b0 = GatedDetectionModel(
        cfg="yolo11n.yaml",
        nc=20,
        baseline_id="B0",
        verbose=False,
    ).to(device)

    torch.manual_seed(42)

    b1 = GatedDetectionModel(
        cfg="yolo11n.yaml",
        nc=20,
        baseline_id="B1",
        target_layer=10,
        channels=256,
        alpha_init=1.0e-3,
        verbose=False,
    ).to(device).train()

    b1.args = get_cfg(cfg=DEFAULT_CFG)
    b1.criterion = None
    b1.zero_grad(set_to_none=True)
    b1.reset_gate_observations()

    required_attributes = (
        "model",
        "stride",
        "names",
        "yaml",
        "loss",
        "load",
    )

    for attribute in required_attributes:
        assert hasattr(b1, attribute), attribute

    assert callable(b1.loss)
    assert callable(b1.load)
    assert b1.model[-1].nc == 20

    b0_parameters = parameter_count(b0)
    b1_parameters = parameter_count(b1)
    added_parameters = b1_parameters - b0_parameters

    batch = make_batch(device)

    loss_components, loss_items = b1(batch)
    total_loss = loss_components.sum()
    total_loss.backward()

    assert b1.gate is not None

    weight_grad = grad_norm(
        b1.gate.linear.weight
    )
    bias_grad = grad_norm(
        b1.gate.linear.bias
    )
    alpha_grad = grad_norm(
        b1.gate.alpha
    )

    gate_state_keys = [
        key
        for key in b1.state_dict()
        if key.startswith("gate.")
    ]

    named_gate_parameters = [
        name
        for name, _ in b1.named_parameters()
        if name.startswith("gate.")
    ]

    print("Model type           :", type(b1).__name__)
    print("Detection classes    :", b1.model[-1].nc)
    print("Detector parameters  :", b0_parameters)
    print("B1 total parameters  :", b1_parameters)
    print("Added parameters     :", added_parameters)
    print("Feature shape        :", b1.last_feature_shape)
    print("Hook calls           :", b1.hook_calls)
    print(
        "Loss components      :",
        loss_components.detach().cpu().tolist(),
    )
    print("Total loss           :", float(total_loss.detach()))
    print(
        "Loss items           :",
        loss_items.detach().cpu().tolist(),
    )
    print("Linear weight grad   :", weight_grad)
    print("Linear bias grad     :", bias_grad)
    print("Alpha grad norm      :", alpha_grad)
    print("Gate state keys      :", len(gate_state_keys))
    print("Named gate parameters:", named_gate_parameters)

    assert b0_parameters == 2593740
    assert added_parameters == 131329
    assert b1.last_feature_shape == (2, 256, 10, 10)
    assert b1.hook_calls == 1

    assert torch.isfinite(loss_components).all()
    assert torch.isfinite(loss_items).all()
    assert float(total_loss.detach()) > 0.0

    assert weight_grad > 0.0
    assert bias_grad > 0.0
    assert alpha_grad > 0.0

    assert gate_state_keys
    assert set(named_gate_parameters) == {
        "gate.alpha",
        "gate.linear.weight",
        "gate.linear.bias",
    }

    b0.close_gate_hook()
    b1.close_gate_hook()

    print("TRAINER-COMPATIBLE B1 MODEL: PASSED")


if __name__ == "__main__":
    main()
