from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from src.qvisionframe.pqfg import PQFG, PQFGConfig


@dataclass(frozen=True)
class HybridPlacementConfig:
    target_layer: int = 10
    channels: int = 256


class YOLOWithPQFG(nn.Module):
    def __init__(
        self,
        yolo_network: nn.Module,
        pqfg_config: PQFGConfig,
        placement: HybridPlacementConfig,
    ) -> None:
        super().__init__()

        if pqfg_config.channels != placement.channels:
            raise ValueError(
                "PQFG channels and placement channels must match"
            )

        self.yolo = yolo_network
        self.placement = placement
        self.pqfg = PQFG(pqfg_config)

        self.hook_calls = 0
        self.last_feature_shape: tuple[int, ...] | None = None

        self._hook_handle = self.yolo.model[
            placement.target_layer
        ].register_forward_hook(self._apply_pqfg)

    def _apply_pqfg(
        self,
        module: nn.Module,
        inputs,
        output: torch.Tensor,
    ) -> torch.Tensor:
        if not torch.is_tensor(output):
            raise TypeError(
                f"Layer {self.placement.target_layer} "
                "must return a tensor"
            )

        if output.shape[1] != self.placement.channels:
            raise RuntimeError(
                f"Expected {self.placement.channels} channels, "
                f"received {output.shape[1]}"
            )

        self.hook_calls += 1
        self.last_feature_shape = tuple(output.shape)

        return self.pqfg(output)

    def forward(self, inputs):
        return self.yolo(inputs)

    def close(self) -> None:
        self._hook_handle.remove()
