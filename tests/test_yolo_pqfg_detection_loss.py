from __future__ import annotations

import torch
from torch import nn
from ultralytics import YOLO
from ultralytics.cfg import get_cfg
from ultralytics.utils import DEFAULT_CFG

from src.qvisionframe.pqfg import PQFG


class YOLOWithPQFG(nn.Module):
    """Inject PQFG after YOLO layer 10 using a forward hook."""

    def __init__(
        self,
        yolo_network: nn.Module,
        target_layer: int = 10,
    ) -> None:
        super().__init__()

        self.yolo = yolo_network
        self.target_layer = target_layer

        self.pqfg = PQFG(
            channels=256,
            n_qubits=4,
            n_layers=2,
            alpha_init=1.0e-3,
        )

        self.hook_calls = 0
        self.last_feature_shape = None

        self._hook_handle = self.yolo.model[
            target_layer
        ].register_forward_hook(self._apply_pqfg)

    def _apply_pqfg(
        self,
        module: nn.Module,
        inputs,
        output: torch.Tensor,
    ) -> torch.Tensor:
        if not torch.is_tensor(output):
            raise TypeError(
                f"Layer {self.target_layer} output is not a tensor"
            )

        self.hook_calls += 1
        self.last_feature_shape = tuple(output.shape)

        return self.pqfg(output)

    def forward(self, batch):
        return self.yolo(batch)

    def close(self) -> None:
        self._hook_handle.remove()


def grad_norm(parameter: torch.Tensor) -> float:
    if parameter.grad is None:
        return 0.0

    return float(
        parameter.grad.detach().float().norm().cpu()
    )


def main() -> None:
    print("===== YOLO + PQFG DETECTION LOSS CHECK =====")

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")

    torch.manual_seed(42)

    device = torch.device("cuda:0")

    wrapper = YOLO("yolo11n.yaml")
    network = wrapper.model.to(device).train()

    # Normalize Ultralytics training/loss configuration.
    network.args = get_cfg(
        cfg=DEFAULT_CFG,
        overrides=wrapper.overrides,
    )
    network.criterion = None

    hybrid = YOLOWithPQFG(
        yolo_network=network,
        target_layer=10,
    )

    # Move classical PQFG components to CUDA.
    hybrid.pqfg.projector.to(device)
    hybrid.pqfg.decoder.to(device)
    hybrid.pqfg.alpha.data = hybrid.pqfg.alpha.data.to(device)

    # lightning.qubit parameters must remain on CPU.
    hybrid.pqfg.quantum_weights.data = (
        hybrid.pqfg.quantum_weights.data.cpu()
    )

    hybrid.zero_grad(set_to_none=True)

    batch_size = 2
    image_size = 320

    images = torch.rand(
        batch_size,
        3,
        image_size,
        image_size,
        dtype=torch.float32,
        device=device,
    )

    # Four objects: two per image.
    classes = torch.tensor(
        [[0.0], [5.0], [2.0], [7.0]],
        dtype=torch.float32,
        device=device,
    )

    # Normalized xywh boxes.
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
            "Expected Ultralytics result "
            "(loss_components, loss_items)"
        )

    loss_components, loss_items = result

    if loss_components.ndim == 0:
        total_loss = loss_components
    else:
        total_loss = loss_components.sum()

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

    yolo_grad_finite = all(
        bool(torch.isfinite(parameter.grad).all())
        for parameter in yolo_parameters_with_grad
    )

    print("GPU                  :", torch.cuda.get_device_name(0))
    print("Hook calls           :", hybrid.hook_calls)
    print("Layer 10 shape       :", hybrid.last_feature_shape)
    print("Loss shape           :", tuple(loss_components.shape))
    print(
        "Loss components      :",
        loss_components.detach().cpu().tolist(),
    )
    print("Total loss           :", float(total_loss.detach()))
    print(
        "Loss items           :",
        loss_items.detach().cpu().tolist(),
    )
    print(
        "Loss finite          :",
        bool(torch.isfinite(loss_components).all()),
    )
    print("YOLO params with grad:", len(yolo_parameters_with_grad))
    print("YOLO grads finite    :", yolo_grad_finite)
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
    assert yolo_grad_finite

    assert hybrid.pqfg.projector[0].weight.grad is not None
    assert hybrid.pqfg.quantum_weights.grad is not None
    assert hybrid.pqfg.decoder.weight.grad is not None
    assert hybrid.pqfg.alpha.grad is not None

    assert projector_grad > 0.0
    assert quantum_grad > 0.0
    assert decoder_grad > 0.0
    assert alpha_grad > 0.0

    hybrid.close()

    print("YOLO + PQFG DETECTION LOSS: PASSED")


if __name__ == "__main__":
    main()
