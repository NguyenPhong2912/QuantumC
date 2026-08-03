from __future__ import annotations

import torch
from ultralytics import YOLO
from ultralytics.cfg import get_cfg
from ultralytics.utils import DEFAULT_CFG

from src.qvisionframe.pqfg import PQFGConfig
from src.qvisionframe.yolo_hybrid import (
    HybridPlacementConfig,
    YOLOWithPQFG,
)


def grad_norm(parameter: torch.Tensor) -> float:
    if parameter.grad is None:
        return 0.0
    return float(parameter.grad.detach().float().norm().cpu())


def main() -> None:
    print("===== OFFICIAL YOLO + PQFG WRAPPER CHECK =====")

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

    pqfg_config = PQFGConfig(
        channels=256,
        n_qubits=4,
        n_layers=2,
        alpha_init=1.0e-3,
        entanglement="ring",
    )

    placement = HybridPlacementConfig(
        target_layer=10,
        channels=256,
    )

    hybrid = YOLOWithPQFG(
        yolo_network=network,
        pqfg_config=pqfg_config,
        placement=placement,
    )

    hybrid.pqfg.projector.to(device)
    hybrid.pqfg.decoder.to(device)
    hybrid.pqfg.alpha.data = hybrid.pqfg.alpha.data.to(device)
    hybrid.pqfg.quantum_weights.data = (
        hybrid.pqfg.quantum_weights.data.cpu()
    )

    hybrid.zero_grad(set_to_none=True)

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

    result = hybrid(batch)

    if not isinstance(result, tuple) or len(result) != 2:
        raise RuntimeError(
            f"Unexpected result type: {type(result).__name__}"
        )

    loss_components, loss_items = result
    total_loss = (
        loss_components
        if loss_components.ndim == 0
        else loss_components.sum()
    )

    total_loss.backward()

    projector_grad = grad_norm(
        hybrid.pqfg.projector[0].weight
    )
    quantum_grad = grad_norm(
        hybrid.pqfg.quantum_weights
    )
    decoder_grad = grad_norm(
        hybrid.pqfg.decoder.weight
    )
    alpha_grad = grad_norm(
        hybrid.pqfg.alpha
    )

    yolo_parameters_with_grad = [
        parameter
        for parameter in hybrid.yolo.parameters()
        if parameter.requires_grad
        and parameter.grad is not None
    ]

    yolo_grads_finite = all(
        bool(torch.isfinite(parameter.grad).all())
        for parameter in yolo_parameters_with_grad
    )

    print("GPU                  :", torch.cuda.get_device_name(0))
    print("Target layer         :", placement.target_layer)
    print("Channels             :", pqfg_config.channels)
    print("Qubits               :", pqfg_config.n_qubits)
    print("Layers               :", pqfg_config.n_layers)
    print("Entanglement         :", pqfg_config.entanglement)
    print("Hook calls           :", hybrid.hook_calls)
    print("Feature shape        :", hybrid.last_feature_shape)
    print(
        "Loss components      :",
        loss_components.detach().cpu().tolist(),
    )
    print("Total loss           :", float(total_loss.detach()))
    print(
        "Loss items           :",
        loss_items.detach().cpu().tolist(),
    )
    print("YOLO params with grad:", len(yolo_parameters_with_grad))
    print("YOLO grads finite    :", yolo_grads_finite)
    print("Projector grad norm  :", projector_grad)
    print("Quantum grad norm    :", quantum_grad)
    print("Decoder grad norm    :", decoder_grad)
    print("Alpha grad norm      :", alpha_grad)
    print(
        "Quantum weight device:",
        hybrid.pqfg.quantum_weights.device,
    )

    assert hybrid.hook_calls == 1
    assert hybrid.last_feature_shape == (2, 256, 10, 10)
    assert torch.isfinite(loss_components).all()
    assert torch.isfinite(loss_items).all()
    assert float(total_loss.detach()) > 0.0

    assert yolo_parameters_with_grad
    assert yolo_grads_finite

    assert projector_grad > 0.0
    assert quantum_grad > 0.0
    assert decoder_grad > 0.0
    assert alpha_grad > 0.0

    assert str(hybrid.pqfg.quantum_weights.device) == "cpu"

    hybrid.close()

    print("OFFICIAL HYBRID WRAPPER: PASSED")


if __name__ == "__main__":
    main()
