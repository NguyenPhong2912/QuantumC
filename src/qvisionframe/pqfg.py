from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import pennylane as qml
import torch
from torch import nn


EntanglementType = Literal["none", "linear", "ring"]


@dataclass(frozen=True)
class PQFGConfig:
    channels: int
    n_qubits: int = 4
    n_layers: int = 2
    hidden_dim: int | None = None
    alpha_init: float = 1.0e-3
    entanglement: EntanglementType = "ring"
    gamma: float = 0.5
    trainable_quantum: bool = True


class PQFG(nn.Module):
    """Parameterized Quantum Feature Gate for BCHW feature tensors."""

    def __init__(self, config: PQFGConfig) -> None:
        super().__init__()

        if config.channels <= 0:
            raise ValueError("channels must be positive")
        if config.n_qubits <= 0:
            raise ValueError("n_qubits must be positive")
        if config.n_layers <= 0:
            raise ValueError("n_layers must be positive")
        if config.entanglement not in {"none", "linear", "ring"}:
            raise ValueError(
                f"Unsupported entanglement: {config.entanglement}"
            )

        self.config = config
        self.channels = config.channels
        self.n_qubits = config.n_qubits
        self.n_layers = config.n_layers
        self.entanglement = config.entanglement
        self.gamma = config.gamma

        hidden_dim = (
            config.hidden_dim
            if config.hidden_dim is not None
            else max(32, config.channels // 4)
        )

        self.projector = nn.Sequential(
            nn.Linear(2 * config.channels, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, config.n_qubits),
        )

        self.quantum_weights = nn.Parameter(
            0.01
            * torch.randn(
                config.n_layers,
                config.n_qubits,
                2,
                dtype=torch.float64,
                device="cpu",
            ),
            requires_grad=config.trainable_quantum,
        )

        self.decoder = nn.Linear(
            config.n_qubits,
            config.channels,
        )

        self.alpha = nn.Parameter(
            torch.tensor(
                config.alpha_init,
                dtype=torch.float32,
            )
        )

        self.qdev = qml.device(
            "lightning.qubit",
            wires=config.n_qubits,
        )

        self.qnode = qml.QNode(
            self._circuit,
            self.qdev,
            interface="torch",
            diff_method="adjoint",
        )

    def _apply_entanglement(self) -> None:
        if self.entanglement == "none":
            return

        for wire in range(self.n_qubits - 1):
            qml.CNOT(wires=[wire, wire + 1])

        if self.entanglement == "ring" and self.n_qubits > 2:
            qml.CNOT(wires=[self.n_qubits - 1, 0])

    def _circuit(
        self,
        angles: torch.Tensor,
        weights: torch.Tensor,
    ):
        for wire in range(self.n_qubits):
            qml.RY(angles[wire], wires=wire)
            qml.RZ(self.gamma * angles[wire], wires=wire)

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

            self._apply_entanglement()

        return [
            qml.expval(qml.PauliZ(wire))
            for wire in range(self.n_qubits)
        ]

    def quantum_forward(
        self,
        angles_cpu: torch.Tensor,
    ) -> torch.Tensor:
        measurements = []

        for sample in angles_cpu:
            sample_measurements = torch.stack(
                self.qnode(
                    sample,
                    self.quantum_weights,
                )
            )
            measurements.append(sample_measurements)

        return torch.stack(measurements, dim=0)

    def forward(
        self,
        features: torch.Tensor,
    ) -> torch.Tensor:
        if features.ndim != 4:
            raise ValueError(
                f"Expected BCHW tensor, received {tuple(features.shape)}"
            )

        if features.shape[1] != self.channels:
            raise ValueError(
                f"Expected {self.channels} channels, "
                f"received {features.shape[1]}"
            )

        z_avg = features.mean(dim=(2, 3))
        z_max = features.amax(dim=(2, 3))
        descriptor = torch.cat([z_avg, z_max], dim=1)

        projected = self.projector(descriptor)
        angles = torch.pi * torch.tanh(projected)

        angles_cpu = angles.to(
            device="cpu",
            dtype=torch.float64,
        )

        measurements_cpu = self.quantum_forward(angles_cpu)

        measurements = measurements_cpu.to(
            device=features.device,
            dtype=self.decoder.weight.dtype,
        )

        gate = torch.sigmoid(
            self.decoder(measurements)
        )

        alpha = self.alpha.to(
            device=features.device,
            dtype=features.dtype,
        )

        return features * (
            1.0 + alpha * gate[:, :, None, None]
        )
