from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class ClassicalMLPGateConfig:
    channels: int
    latent_dim: int = 4
    hidden_dim: int | None = None
    transform_hidden_dim: int = 8
    alpha_init: float = 1.0e-3


class ClassicalMLPGate(nn.Module):
    """
    Classical control with the same tensor-to-gate interface as PQFG.

    BCHW feature
      -> GAP/GMP descriptor
      -> projector
      -> latent MLP transform
      -> channel decoder
      -> residual modulation
    """

    def __init__(
        self,
        config: ClassicalMLPGateConfig,
    ) -> None:
        super().__init__()

        if config.channels <= 0:
            raise ValueError("channels must be positive")
        if config.latent_dim <= 0:
            raise ValueError("latent_dim must be positive")
        if config.transform_hidden_dim <= 0:
            raise ValueError(
                "transform_hidden_dim must be positive"
            )

        self.config = config
        self.channels = config.channels
        self.latent_dim = config.latent_dim

        hidden_dim = (
            config.hidden_dim
            if config.hidden_dim is not None
            else max(32, config.channels // 4)
        )

        # Same descriptor projector used by the PQFG pilot.
        self.projector = nn.Sequential(
            nn.Linear(
                2 * config.channels,
                hidden_dim,
            ),
            nn.GELU(),
            nn.Linear(
                hidden_dim,
                config.latent_dim,
            ),
        )

        # Classical replacement for the variational circuit.
        self.latent_transform = nn.Sequential(
            nn.Linear(
                config.latent_dim,
                config.transform_hidden_dim,
            ),
            nn.Tanh(),
            nn.Linear(
                config.transform_hidden_dim,
                config.latent_dim,
            ),
            nn.Tanh(),
        )

        # Same measurement/latent-to-channel interface.
        self.decoder = nn.Linear(
            config.latent_dim,
            config.channels,
        )

        self.alpha = nn.Parameter(
            torch.tensor(
                config.alpha_init,
                dtype=torch.float32,
            )
        )

    def forward(
        self,
        features: torch.Tensor,
    ) -> torch.Tensor:
        if features.ndim != 4:
            raise ValueError(
                "Expected BCHW feature tensor, "
                f"received shape {tuple(features.shape)}"
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

        latent = self.projector(descriptor)
        transformed = self.latent_transform(latent)

        gate = torch.sigmoid(
            self.decoder(transformed)
        )

        alpha = self.alpha.to(
            device=features.device,
            dtype=features.dtype,
        )

        return features * (
            1.0 + alpha * gate[:, :, None, None]
        )


@dataclass(frozen=True)
class ParameterMatchedGateConfig:
    channels: int
    latent_dim: int = 4
    n_layers: int = 2
    hidden_dim: int | None = None
    alpha_init: float = 1.0e-3


class ElementwiseAffineLayer(nn.Module):
    """Elementwise affine transformation with 2d parameters."""

    def __init__(self, dimension: int) -> None:
        super().__init__()

        self.weight = nn.Parameter(
            torch.ones(dimension, dtype=torch.float32)
        )
        self.bias = nn.Parameter(
            torch.zeros(dimension, dtype=torch.float32)
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return torch.tanh(
            inputs * self.weight + self.bias
        )


class ParameterMatchedClassicalGate(nn.Module):
    """
    Classical control matched to PQFG's trainable parameter count.

    The projector, decoder and residual scalar are identical in size
    to PQFG. Each elementwise affine layer contributes 2q parameters,
    matching the two rotation parameters per qubit in one PQC layer.
    """

    def __init__(
        self,
        config: ParameterMatchedGateConfig,
    ) -> None:
        super().__init__()

        if config.channels <= 0:
            raise ValueError("channels must be positive")
        if config.latent_dim <= 0:
            raise ValueError("latent_dim must be positive")
        if config.n_layers <= 0:
            raise ValueError("n_layers must be positive")

        self.config = config
        self.channels = config.channels
        self.latent_dim = config.latent_dim
        self.n_layers = config.n_layers

        hidden_dim = (
            config.hidden_dim
            if config.hidden_dim is not None
            else max(32, config.channels // 4)
        )

        self.projector = nn.Sequential(
            nn.Linear(2 * config.channels, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, config.latent_dim),
        )

        self.latent_transform = nn.Sequential(
            *[
                ElementwiseAffineLayer(config.latent_dim)
                for _ in range(config.n_layers)
            ]
        )

        self.decoder = nn.Linear(
            config.latent_dim,
            config.channels,
        )

        self.alpha = nn.Parameter(
            torch.tensor(
                config.alpha_init,
                dtype=torch.float32,
            )
        )

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

        latent = self.projector(descriptor)
        transformed = self.latent_transform(latent)
        gate = torch.sigmoid(self.decoder(transformed))

        alpha = self.alpha.to(
            device=features.device,
            dtype=features.dtype,
        )

        return features * (
            1.0 + alpha * gate[:, :, None, None]
        )


@dataclass(frozen=True)
class TrigonometricGateConfig:
    channels: int
    latent_dim: int = 4
    hidden_dim: int | None = None
    alpha_init: float = 1.0e-3


class TrigonometricClassicalGate(nn.Module):
    """
    Classical trigonometric control:
    descriptor -> projector -> sin/cos transform -> decoder -> residual gate.
    """

    def __init__(
        self,
        config: TrigonometricGateConfig,
    ) -> None:
        super().__init__()

        if config.channels <= 0:
            raise ValueError("channels must be positive")
        if config.latent_dim <= 0:
            raise ValueError("latent_dim must be positive")

        self.config = config
        self.channels = config.channels
        self.latent_dim = config.latent_dim

        hidden_dim = (
            config.hidden_dim
            if config.hidden_dim is not None
            else max(32, config.channels // 4)
        )

        self.projector = nn.Sequential(
            nn.Linear(2 * config.channels, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, config.latent_dim),
        )

        self.sin_transform = nn.Linear(
            config.latent_dim,
            config.latent_dim,
        )

        self.cos_transform = nn.Linear(
            config.latent_dim,
            config.latent_dim,
        )

        self.decoder = nn.Linear(
            config.latent_dim,
            config.channels,
        )

        self.alpha = nn.Parameter(
            torch.tensor(
                config.alpha_init,
                dtype=torch.float32,
            )
        )

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

        latent = self.projector(descriptor)

        transformed = 0.5 * (
            torch.sin(self.sin_transform(latent))
            + torch.cos(self.cos_transform(latent))
        )

        gate = torch.sigmoid(
            self.decoder(transformed)
        )

        alpha = self.alpha.to(
            device=features.device,
            dtype=features.dtype,
        )

        return features * (
            1.0 + alpha * gate[:, :, None, None]
        )


@dataclass(frozen=True)
class LinearGateConfig:
    channels: int
    alpha_init: float = 1.0e-3


class LinearClassicalGate(nn.Module):
    """
    Simple GAP/GMP linear channel gate.

    BCHW feature
      -> GAP and GMP
      -> concatenate to 2C descriptor
      -> linear projection to C channels
      -> sigmoid
      -> residual channel modulation
    """

    def __init__(
        self,
        config: LinearGateConfig,
    ) -> None:
        super().__init__()

        if config.channels <= 0:
            raise ValueError("channels must be positive")

        self.config = config
        self.channels = config.channels

        self.linear = nn.Linear(
            2 * config.channels,
            config.channels,
        )

        self.alpha = nn.Parameter(
            torch.tensor(
                config.alpha_init,
                dtype=torch.float32,
            )
        )

    def forward(
        self,
        features: torch.Tensor,
    ) -> torch.Tensor:
        if features.ndim != 4:
            raise ValueError(
                "Expected BCHW feature tensor, "
                f"received shape {tuple(features.shape)}"
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

        gate = torch.sigmoid(
            self.linear(descriptor)
        )

        alpha = self.alpha.to(
            device=features.device,
            dtype=features.dtype,
        )

        return features * (
            1.0 + alpha * gate[:, :, None, None]
        )


@dataclass(frozen=True)
class ChannelAttentionGateConfig:
    channels: int
    reduction: int = 16
    alpha_init: float = 1.0e-3


class ChannelAttentionClassicalGate(nn.Module):
    """
    SE/CBAM-style shared channel-attention gate.

    GAP(F) and GMP(F) are passed independently through the same
    two-layer bottleneck MLP. Their logits are summed before sigmoid.
    """

    def __init__(
        self,
        config: ChannelAttentionGateConfig,
    ) -> None:
        super().__init__()

        if config.channels <= 0:
            raise ValueError("channels must be positive")

        if config.reduction <= 0:
            raise ValueError("reduction must be positive")

        hidden_channels = max(
            1,
            config.channels // config.reduction,
        )

        self.config = config
        self.channels = config.channels
        self.hidden_channels = hidden_channels

        self.shared_mlp = nn.Sequential(
            nn.Linear(
                config.channels,
                hidden_channels,
            ),
            nn.ReLU(inplace=False),
            nn.Linear(
                hidden_channels,
                config.channels,
            ),
        )

        self.alpha = nn.Parameter(
            torch.tensor(
                config.alpha_init,
                dtype=torch.float32,
            )
        )

    def forward(
        self,
        features: torch.Tensor,
    ) -> torch.Tensor:
        if features.ndim != 4:
            raise ValueError(
                "Expected BCHW feature tensor, "
                f"received {tuple(features.shape)}"
            )

        if features.shape[1] != self.channels:
            raise ValueError(
                f"Expected {self.channels} channels, "
                f"received {features.shape[1]}"
            )

        z_avg = features.mean(dim=(2, 3))
        z_max = features.amax(dim=(2, 3))

        logits = (
            self.shared_mlp(z_avg)
            + self.shared_mlp(z_max)
        )

        gate = torch.sigmoid(logits)

        alpha = self.alpha.to(
            device=features.device,
            dtype=features.dtype,
        )

        return features * (
            1.0 + alpha * gate[:, :, None, None]
        )
