from __future__ import annotations

import torch
from torch import nn

from src.qvisionframe.classical_gate import (
    ParameterMatchedClassicalGate,
    ParameterMatchedGateConfig,
)
from src.qvisionframe.yolo_hybrid import HybridPlacementConfig


class YOLOWithParameterMatchedGate(nn.Module):
    """Insert an exactly parameter-matched classical gate into YOLO."""

    def __init__(
        self,
        yolo_network: nn.Module,
        gate_config: ParameterMatchedGateConfig,
        placement: HybridPlacementConfig,
    ) -> None:
        super().__init__()

        if gate_config.channels != placement.channels:
            raise ValueError(
                "Gate and placement channels must match"
            )

        self.yolo = yolo_network
        self.placement = placement
        self.gate = ParameterMatchedClassicalGate(
            gate_config
        )

        self.hook_calls = 0
        self.last_feature_shape: tuple[int, ...] | None = None

        self._hook_handle = self.yolo.model[
            placement.target_layer
        ].register_forward_hook(self._apply_gate)

    def _apply_gate(
        self,
        module: nn.Module,
        inputs,
        output: torch.Tensor,
    ) -> torch.Tensor:
        if not torch.is_tensor(output):
            raise TypeError("Target layer must return a tensor")

        if output.ndim != 4:
            raise RuntimeError(
                f"Expected BCHW output, received {tuple(output.shape)}"
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
