from __future__ import annotations

import torch
from ultralytics import YOLO
from ultralytics.cfg import get_cfg
from ultralytics.utils import DEFAULT_CFG

from src.qvisionframe.config import load_config
from src.qvisionframe.factory import build_hybrid_model


def grad_norm(parameter: torch.Tensor) -> float:
    if parameter.grad is None:
        return 0.0
    return float(parameter.grad.detach().float().norm().cpu())


def main() -> None:
    print("===== FROZEN RANDOM PQC CHECK =====")

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")

    torch.manual_seed(42)
    device = torch.device("cuda:0")

    config = load_config(
        "configs/experiments/pqfg_p5_q4_l2_ring_frozen.yaml"
    )

    model = build_hybrid_model(
        config=config,
        training=True,
    )

    # Ensure training loss configuration is valid.
    if not hasattr(model.yolo.args, "box"):
        wrapper = YOLO("yolo11n.yaml")
        model.yolo.args = get_cfg(
            cfg=DEFAULT_CFG,
            overrides=wrapper.overrides,
        )
        model.yolo.criterion = None

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

    quantum_before = (
        model.pqfg.quantum_weights.detach().clone()
    )

    loss_components, loss_items = model(batch)
    total_loss = loss_components.sum()
    total_loss.backward()

    projector_grad = grad_norm(
        model.pqfg.projector[0].weight
    )
    decoder_grad = grad_norm(
        model.pqfg.decoder.weight
    )
    alpha_grad = grad_norm(
        model.pqfg.alpha
    )

    quantum_after = (
        model.pqfg.quantum_weights.detach().clone()
    )

    print("Experiment            :", config.experiment.name)
    print("Feature shape         :", model.last_feature_shape)
    print("Hook calls            :", model.hook_calls)
    print(
        "Quantum trainable     :",
        model.pqfg.quantum_weights.requires_grad,
    )
    print(
        "Quantum gradient      :",
        model.pqfg.quantum_weights.grad,
    )
    print(
        "Quantum weights equal :",
        bool(torch.equal(quantum_before, quantum_after)),
    )
    print(
        "Loss components       :",
        loss_components.detach().cpu().tolist(),
    )
    print("Total loss            :", float(total_loss.detach()))
    print(
        "Loss finite           :",
        bool(torch.isfinite(loss_components).all()),
    )
    print("Projector grad norm   :", projector_grad)
    print("Decoder grad norm     :", decoder_grad)
    print("Alpha grad norm       :", alpha_grad)

    assert model.hook_calls == 1
    assert model.last_feature_shape == (2, 256, 10, 10)

    assert model.pqfg.quantum_weights.requires_grad is False
    assert model.pqfg.quantum_weights.grad is None
    assert torch.equal(quantum_before, quantum_after)

    assert torch.isfinite(loss_components).all()
    assert torch.isfinite(loss_items).all()
    assert float(total_loss.detach()) > 0.0

    assert projector_grad > 0.0
    assert decoder_grad > 0.0
    assert alpha_grad > 0.0

    model.close()

    print("FROZEN RANDOM PQC    : PASSED")


if __name__ == "__main__":
    main()
