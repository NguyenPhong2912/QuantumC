from __future__ import annotations

import torch
from torch import nn
from ultralytics import YOLO

from src.qvisionframe.pqfg import PQFG


class YOLOWithPQFG(nn.Module):
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
        self.last_input_shape = None
        self.last_output_shape = None

        target = self.yolo.model[target_layer]

        self._hook_handle = target.register_forward_hook(
            self._apply_pqfg
        )

    def _apply_pqfg(
        self,
        module: nn.Module,
        inputs,
        output: torch.Tensor,
    ) -> torch.Tensor:
        if not torch.is_tensor(output):
            raise TypeError(
                f"Layer {self.target_layer} output must be a tensor"
            )

        self.hook_calls += 1
        self.last_input_shape = tuple(output.shape)

        modulated = self.pqfg(output)

        self.last_output_shape = tuple(modulated.shape)
        return modulated

    def forward(self, x):
        return self.yolo(x)

    def close(self) -> None:
        self._hook_handle.remove()


def grad_norm(parameter: torch.Tensor) -> float:
    if parameter.grad is None:
        return 0.0
    return float(parameter.grad.detach().norm().cpu())


def main() -> None:
    print("===== YOLO + PQFG INJECTION CHECK =====")

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")

    torch.manual_seed(42)

    device = torch.device("cuda:0")

    base = YOLO("yolo11n.yaml").model.to(device).train()
    hybrid = YOLOWithPQFG(base).to(device)

    # quantum_weights must remain on CPU for lightning.qubit.
    hybrid.pqfg.quantum_weights.data = (
        hybrid.pqfg.quantum_weights.data.cpu()
    )

    x = torch.randn(
        1,
        3,
        320,
        320,
        dtype=torch.float32,
        device=device,
        requires_grad=True,
    )

    hybrid.zero_grad(set_to_none=True)

    outputs = hybrid(x)

    tensors = []

    def collect(obj):
        if torch.is_tensor(obj):
            tensors.append(obj)
        elif isinstance(obj, dict):
            for value in obj.values():
                collect(value)
        elif isinstance(obj, (list, tuple)):
            for value in obj:
                collect(value)

    collect(outputs)

    differentiable = [
        tensor
        for tensor in tensors
        if tensor.is_floating_point() and tensor.requires_grad
    ]

    if not differentiable:
        raise RuntimeError("No differentiable YOLO output found")

    loss = sum(
        tensor.float().square().mean()
        for tensor in differentiable
    )

    loss.backward()

    print("GPU                  :", torch.cuda.get_device_name(0))
    print("Hook calls           :", hybrid.hook_calls)
    print("Layer 10 input shape :", hybrid.last_input_shape)
    print("PQFG output shape    :", hybrid.last_output_shape)
    print("Loss                 :", float(loss.detach()))
    print(
        "Projector device     :",
        hybrid.pqfg.projector[0].weight.device,
    )
    print(
        "Quantum device       :",
        hybrid.pqfg.quantum_weights.device,
    )
    print(
        "Decoder device       :",
        hybrid.pqfg.decoder.weight.device,
    )
    print(
        "Projector grad norm  :",
        grad_norm(hybrid.pqfg.projector[0].weight),
    )
    print(
        "Quantum grad norm    :",
        grad_norm(hybrid.pqfg.quantum_weights),
    )
    print(
        "Decoder grad norm    :",
        grad_norm(hybrid.pqfg.decoder.weight),
    )
    print(
        "Alpha grad norm      :",
        grad_norm(hybrid.pqfg.alpha),
    )
    print(
        "Input grad norm      :",
        grad_norm(x),
    )

    assert hybrid.hook_calls == 1
    assert hybrid.last_input_shape == (1, 256, 10, 10)
    assert hybrid.last_output_shape == (1, 256, 10, 10)

    assert hybrid.pqfg.projector[0].weight.grad is not None
    assert hybrid.pqfg.quantum_weights.grad is not None
    assert hybrid.pqfg.decoder.weight.grad is not None
    assert hybrid.pqfg.alpha.grad is not None
    assert x.grad is not None

    assert grad_norm(hybrid.pqfg.projector[0].weight) > 0.0
    assert grad_norm(hybrid.pqfg.quantum_weights) > 0.0
    assert grad_norm(hybrid.pqfg.decoder.weight) > 0.0
    assert grad_norm(hybrid.pqfg.alpha) > 0.0
    assert grad_norm(x) > 0.0

    hybrid.close()

    print("YOLO + PQFG INJECTION: PASSED")


if __name__ == "__main__":
    main()
