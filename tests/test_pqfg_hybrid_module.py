import time

import pennylane as qml
import torch
from torch import nn


torch.manual_seed(42)

BATCH_SIZE = 3
CHANNELS = 16
HEIGHT = 8
WIDTH = 8
N_QUBITS = 4
N_LAYERS = 2


class PQFG(nn.Module):
    """Minimal Parameterized Quantum Feature Gate."""

    def __init__(
        self,
        channels: int,
        n_qubits: int = 4,
        n_layers: int = 2,
    ) -> None:
        super().__init__()

        if channels <= 0:
            raise ValueError("channels must be positive")

        if n_qubits <= 0:
            raise ValueError("n_qubits must be positive")

        self.channels = channels
        self.n_qubits = n_qubits
        self.n_layers = n_layers

        hidden_dim = max(8, channels // 2)

        # [GAP; GMP] has dimension 2C.
        self.projector = nn.Sequential(
            nn.Linear(2 * channels, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, n_qubits),
        ).double()

        self.quantum_weights = nn.Parameter(
            0.01
            * torch.randn(
                n_layers,
                n_qubits,
                2,
                dtype=torch.float64,
            )
        )

        # Decode q measurements back to C channel gates.
        self.decoder = nn.Linear(
            n_qubits,
            channels,
        ).double()

        # Near-identity residual initialization.
        self.alpha = nn.Parameter(
            torch.tensor(
                1.0e-3,
                dtype=torch.float64,
            )
        )

        self.qdev = qml.device(
            "lightning.qubit",
            wires=n_qubits,
        )

        self.qnode = qml.QNode(
            self._circuit,
            self.qdev,
            interface="torch",
            diff_method="adjoint",
        )

    def _circuit(
        self,
        angles: torch.Tensor,
        weights: torch.Tensor,
    ):
        # Angle encoding.
        for wire in range(self.n_qubits):
            qml.RY(angles[wire], wires=wire)
            qml.RZ(0.5 * angles[wire], wires=wire)

        # Variational circuit with ring entanglement.
        for layer in range(self.n_layers):
            for wire in range(self.n_qubits):
                qml.RZ(
                    weights[layer, wire, 0],
                    wires=wire,
                )
                qml.RY(
                    weights[layer, wire, 1],
                    wires=wire,
                )

            for wire in range(self.n_qubits - 1):
                qml.CNOT(wires=[wire, wire + 1])

            qml.CNOT(
                wires=[self.n_qubits - 1, 0]
            )

        return [
            qml.expval(qml.PauliZ(wire))
            for wire in range(self.n_qubits)
        ]

    def quantum_forward(
        self,
        angles: torch.Tensor,
    ) -> torch.Tensor:
        # Explicit per-sample execution is intentionally used for
        # this correctness test. Vectorization is optimized later.
        outputs = []

        for sample in angles:
            measured = torch.stack(
                self.qnode(
                    sample,
                    self.quantum_weights,
                )
            )
            outputs.append(measured)

        return torch.stack(outputs, dim=0)

    def forward(
        self,
        features: torch.Tensor,
    ):
        if features.ndim != 4:
            raise ValueError(
                "Expected features with shape [B, C, H, W]"
            )

        if features.shape[1] != self.channels:
            raise ValueError(
                f"Expected {self.channels} channels, "
                f"received {features.shape[1]}"
            )

        z_avg = features.mean(dim=(2, 3))
        z_max = features.amax(dim=(2, 3))
        descriptor = torch.cat(
            [z_avg, z_max],
            dim=1,
        )

        projected = self.projector(descriptor)

        # Equation: u_tilde = pi * tanh(u).
        angles = torch.pi * torch.tanh(projected)

        measurements = self.quantum_forward(angles)

        # PennyLane may return measurements in float32 even when
        # the surrounding PyTorch module uses float64.
        measurements = measurements.to(
            dtype=self.decoder.weight.dtype,
            device=self.decoder.weight.device,
        )

        gate = torch.sigmoid(
            self.decoder(measurements)
        )

        output = features * (
            1.0
            + self.alpha
            * gate[:, :, None, None]
        )

        diagnostics = {
            "descriptor": descriptor,
            "angles": angles,
            "measurements": measurements,
            "gate": gate,
        }

        return output, diagnostics


def parameter_gradient_norm(
    parameter: torch.Tensor,
) -> float:
    if parameter.grad is None:
        return 0.0

    return float(
        parameter.grad.detach().norm().cpu()
    )


def main() -> None:
    model = PQFG(
        channels=CHANNELS,
        n_qubits=N_QUBITS,
        n_layers=N_LAYERS,
    )

    features = torch.randn(
        BATCH_SIZE,
        CHANNELS,
        HEIGHT,
        WIDTH,
        dtype=torch.float64,
        requires_grad=True,
    )

    start = time.perf_counter()

    output, diagnostics = model(features)

    # A simple differentiable objective for environment validation.
    loss = (
        output.square().mean()
        + 0.01
        * diagnostics["measurements"].square().mean()
    )

    loss.backward()

    elapsed = time.perf_counter() - start

    projector_grad = parameter_gradient_norm(
        model.projector[0].weight
    )
    quantum_grad = parameter_gradient_norm(
        model.quantum_weights
    )
    decoder_grad = parameter_gradient_norm(
        model.decoder.weight
    )
    alpha_grad = parameter_gradient_norm(
        model.alpha
    )
    feature_grad = parameter_gradient_norm(
        features
    )

    print("===== HYBRID PQFG MODULE CHECK =====")
    print("Input shape           :", tuple(features.shape))
    print("Descriptor shape      :", tuple(diagnostics["descriptor"].shape))
    print("Angle shape           :", tuple(diagnostics["angles"].shape))
    print("Measurement shape     :", tuple(diagnostics["measurements"].shape))
    print("Gate shape            :", tuple(diagnostics["gate"].shape))
    print("Output shape          :", tuple(output.shape))
    print("Loss                  :", float(loss.detach()))
    print("Alpha                 :", float(model.alpha.detach()))
    print("Projector grad norm   :", projector_grad)
    print("Quantum grad norm     :", quantum_grad)
    print("Decoder grad norm     :", decoder_grad)
    print("Alpha grad norm       :", alpha_grad)
    print("Feature grad norm     :", feature_grad)
    print("Output finite         :", bool(torch.isfinite(output).all()))
    print(
        "Measurements finite  :",
        bool(torch.isfinite(diagnostics["measurements"]).all()),
    )
    print("Elapsed seconds       :", round(elapsed, 6))

    assert output.shape == features.shape
    assert diagnostics["descriptor"].shape == (
        BATCH_SIZE,
        2 * CHANNELS,
    )
    assert diagnostics["angles"].shape == (
        BATCH_SIZE,
        N_QUBITS,
    )
    assert diagnostics["measurements"].shape == (
        BATCH_SIZE,
        N_QUBITS,
    )
    assert diagnostics["gate"].shape == (
        BATCH_SIZE,
        CHANNELS,
    )

    assert torch.isfinite(output).all()
    assert torch.isfinite(
        diagnostics["measurements"]
    ).all()

    assert features.grad is not None
    assert model.quantum_weights.grad is not None
    assert model.decoder.weight.grad is not None
    assert model.projector[0].weight.grad is not None
    assert model.alpha.grad is not None

    assert feature_grad > 0.0
    assert quantum_grad > 0.0
    assert decoder_grad > 0.0
    assert projector_grad > 0.0
    assert alpha_grad > 0.0

    print("HYBRID PQFG TEST      : PASSED")


if __name__ == "__main__":
    main()
