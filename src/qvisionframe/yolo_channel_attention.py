from __future__ import annotations

import torch
from torch import nn

from src.qvisionframe.classical_gate import (
    ChannelAttentionClassicalGate,
    ChannelAttentionGateConfig,
)
from src.qvisionframe.yolo_hybrid import HybridPlacementConfig


class YOLOWithChannelAttentionGate(nn.Module):
    """Insert the B2 SE/CBAM-style channel gate into YOLO."""

    def __init__(
        self,
        yolo_network: nn.Module,
        gate_config: ChannelAttentionGateConfig,
        placement: HybridPlacementConfig,
    ) -> None:
        super().__init__()

        if gate_config.channels != placement.channels:
            raise ValueError(
                "Gate channels and placement channels must match"
            )

        if not hasattr(yolo_network, "model"):
            raise TypeError(
                "yolo_network must expose layers through '.model'"
            )

        if not 0 <= placement.target_layer < len(yolo_network.model):
            raise IndexError(
                f"Invalid target layer {placement.target_layer}"
            )

        self.yolo = yolo_network
        self.placement = placement
        self.gate = ChannelAttentionClassicalGate(
            gate_config
        )

        self.hook_calls = 0
        self.last_feature_shape: tuple[int, ...] | None = None

        target_module = self.yolo.model[
            placement.target_layer
        ]

        self._hook_handle = target_module.register_forward_hook(
            self._apply_gate
        )

    def _apply_gate(
        self,
        module: nn.Module,
        inputs,
        output: torch.Tensor,
    ) -> torch.Tensor:
        if not torch.is_tensor(output):
            raise TypeError(
                "Target layer must return a tensor"
            )

        if output.ndim != 4:
            raise RuntimeError(
                "Expected BCHW output, received "
                f"{tuple(output.shape)}"
            )

        if output.shape[1] != self.placement.channels:
            raise RuntimeError(
                f"Expected {self.placement.channels} channels, "
                f"received {output.shape[1]}"
            )

        self.hook_calls += 1
        self.last_feature_shape = tuple(output.shape)

        return self.gate(output)

    def forward(self, inputs):
        return self.yolo(inputs)

    def close(self) -> None:
        if self._hook_handle is not None:
            self._hook_handle.remove()
            self._hook_handle = None
