from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import torch
from torch import nn
from ultralytics.nn.tasks import DetectionModel

from src.qvisionframe.classical_gate import (
    ChannelAttentionClassicalGate,
    ChannelAttentionGateConfig,
    ClassicalMLPGate,
    ClassicalMLPGateConfig,
    LinearClassicalGate,
    LinearGateConfig,
    ParameterMatchedClassicalGate,
    ParameterMatchedGateConfig,
    TrigonometricClassicalGate,
    TrigonometricGateConfig,
)


ClassicalBaselineID = Literal[
    "B0",
    "B1",
    "B2",
    "B3",
    "B4",
    "B5",
]


class GatedDetectionModel(DetectionModel):
    """
    Ultralytics DetectionModel with an internally registered
    feature gate at a selected detector layer.

    Supported baselines:
      B0: plain detector
      B1: linear gate
      B2: channel-attention gate
      B3: classical MLP gate
      B4: parameter-matched classical gate
      B5: trigonometric classical gate
    """

    SUPPORTED_BASELINES = {
        "B0",
        "B1",
        "B2",
        "B3",
        "B4",
        "B5",
    }

    def __init__(
        self,
        cfg: str | Path | dict[str, Any] = "yolo11n.yaml",
        *,
        nc: int = 80,
        baseline_id: ClassicalBaselineID = "B0",
        target_layer: int = 10,
        channels: int = 256,
        latent_dim: int = 4,
        hidden_dim: int | None = 64,
        transform_hidden_dim: int = 8,
        n_layers: int = 2,
        reduction: int = 16,
        alpha_init: float = 1.0e-3,
        verbose: bool = True,
    ) -> None:
        if baseline_id not in self.SUPPORTED_BASELINES:
            raise ValueError(
                "Unsupported classical baseline "
                f"{baseline_id!r}; expected one of "
                f"{sorted(self.SUPPORTED_BASELINES)}"
            )

        if target_layer < 0:
            raise ValueError(
                "target_layer must be non-negative"
            )

        if channels <= 0:
            raise ValueError(
                "channels must be positive"
            )

        if latent_dim <= 0:
            raise ValueError(
                "latent_dim must be positive"
            )

        if hidden_dim is not None and hidden_dim <= 0:
            raise ValueError(
                "hidden_dim must be positive or None"
            )

        if transform_hidden_dim <= 0:
            raise ValueError(
                "transform_hidden_dim must be positive"
            )

        if n_layers <= 0:
            raise ValueError(
                "n_layers must be positive"
            )

        if reduction <= 0:
            raise ValueError(
                "reduction must be positive"
            )

        super().__init__(
            cfg=cfg,
            ch=3,
            nc=nc,
            verbose=verbose,
        )

        if target_layer >= len(self.model):
            raise IndexError(
                f"Invalid target layer {target_layer}; "
                f"detector contains {len(self.model)} layers"
            )

        self.baseline_id = baseline_id
        self.target_layer = target_layer
        self.gate_channels = channels

        # Store explicit architecture metadata so the checkpoint
        # documents the exact gate configuration.
        self.qvf_gate_config = {
            "baseline_id": baseline_id,
            "target_layer": target_layer,
            "channels": channels,
            "latent_dim": latent_dim,
            "hidden_dim": hidden_dim,
            "transform_hidden_dim": transform_hidden_dim,
            "n_layers": n_layers,
            "reduction": reduction,
            "alpha_init": alpha_init,
        }

        self.gate = self._build_gate(
            baseline_id=baseline_id,
            channels=channels,
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
            transform_hidden_dim=transform_hidden_dim,
            n_layers=n_layers,
            reduction=reduction,
            alpha_init=alpha_init,
        )

        self.hook_calls = 0
        self.last_feature_shape: (
            tuple[int, ...] | None
        ) = None
        self._gate_hook_handle = None

        if self.gate is not None:
            self._gate_hook_handle = self.model[
                target_layer
            ].register_forward_hook(
                self._apply_feature_gate
            )

    @staticmethod
    def _build_gate(
        *,
        baseline_id: ClassicalBaselineID,
        channels: int,
        latent_dim: int,
        hidden_dim: int | None,
        transform_hidden_dim: int,
        n_layers: int,
        reduction: int,
        alpha_init: float,
    ) -> nn.Module | None:
        if baseline_id == "B0":
            return None

        if baseline_id == "B1":
            return LinearClassicalGate(
                LinearGateConfig(
                    channels=channels,
                    alpha_init=alpha_init,
                )
            )

        if baseline_id == "B2":
            return ChannelAttentionClassicalGate(
                ChannelAttentionGateConfig(
                    channels=channels,
                    reduction=reduction,
                    alpha_init=alpha_init,
                )
            )

        if baseline_id == "B3":
            return ClassicalMLPGate(
                ClassicalMLPGateConfig(
                    channels=channels,
                    latent_dim=latent_dim,
                    hidden_dim=hidden_dim,
                    transform_hidden_dim=(
                        transform_hidden_dim
                    ),
                    alpha_init=alpha_init,
                )
            )

        if baseline_id == "B4":
            return ParameterMatchedClassicalGate(
                ParameterMatchedGateConfig(
                    channels=channels,
                    latent_dim=latent_dim,
                    n_layers=n_layers,
                    hidden_dim=hidden_dim,
                    alpha_init=alpha_init,
                )
            )

        if baseline_id == "B5":
            return TrigonometricClassicalGate(
                TrigonometricGateConfig(
                    channels=channels,
                    latent_dim=latent_dim,
                    hidden_dim=hidden_dim,
                    alpha_init=alpha_init,
                )
            )

        raise AssertionError(
            f"Unhandled baseline {baseline_id}"
        )

    def _apply_feature_gate(
        self,
        module: nn.Module,
        inputs: tuple[Any, ...],
        output: torch.Tensor,
    ) -> torch.Tensor:
        if self.gate is None:
            return output

        if not torch.is_tensor(output):
            raise TypeError(
                f"Layer {self.target_layer} returned "
                f"{type(output).__name__}, expected Tensor"
            )

        if output.ndim != 4:
            raise RuntimeError(
                "Feature gate requires BCHW output, "
                f"received {tuple(output.shape)}"
            )

        if output.shape[1] != self.gate_channels:
            raise RuntimeError(
                f"Expected {self.gate_channels} channels "
                f"at layer {self.target_layer}, received "
                f"{output.shape[1]}"
            )

        self.hook_calls += 1
        self.last_feature_shape = tuple(output.shape)

        return self.gate(output)

    def reset_gate_observations(self) -> None:
        self.hook_calls = 0
        self.last_feature_shape = None

    def close_gate_hook(self) -> None:
        if self._gate_hook_handle is not None:
            self._gate_hook_handle.remove()
            self._gate_hook_handle = None
