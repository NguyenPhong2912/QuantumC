from __future__ import annotations

import torch
from torch import nn

from src.qvisionframe.classical_gate import (
    ClassicalMLPGate,
    ClassicalMLPGateConfig,
)
from src.qvisionframe.yolo_hybrid import HybridPlacementConfig


class YOLOWithClassicalGate(nn.Module):
    """Insert a Classical MLP Gate at a selected YOLO layer."""

    def __init__(
        self,
        yolo_network: nn.Module,
        gate_config: ClassicalMLPGateConfig,
        placement: HybridPlacementConfig,
    ) -> None:
        super().__init__()

        if gate_config.channels != placement.channels:
            raise ValueError(
                "Gate channels and placement channels must match: "
                f"{gate_config.channels} != {placement.channels}"
            )

        if not hasattr(yolo_network, "model"):
            raise TypeError(
                "yolo_network must expose its layers through '.model'"
            )

        if not 0 <= placement.target_layer < len(yolo_network.model):
            raise IndexError(
                f"Invalid target layer {placement.target_layer}; "
                f"YOLO has {len(yolo_network.model)} layers"
            )

        self.yolo = yolo_network
        self.placement = placement
        self.gate = ClassicalMLPGate(gate_config)

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
                f"Layer {self.placement.target_layer} must return "
                f"a tensor, received {type(output).__name__}"
            )

        if output.ndim != 4:
            raise RuntimeError(
                "Classical gate expects BCHW features, received "
                f"shape {tuple(output.shape)}"
            )

        if output.shape[1] != self.placement.channels:
            raise RuntimeError(
                f"Expected {self.placement.channels} channels at "
                f"layer {self.placement.target_layer}, received "
                f"{output.shape[1]}"
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
