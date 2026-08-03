from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
from torch import nn
from ultralytics import YOLO

from src.qvisionframe.classical_gate import (
    ChannelAttentionGateConfig,
    ClassicalMLPGateConfig,
    LinearGateConfig,
    ParameterMatchedGateConfig,
    TrigonometricGateConfig,
)
from src.qvisionframe.pqfg import PQFGConfig
from src.qvisionframe.yolo_channel_attention import (
    YOLOWithChannelAttentionGate,
)
from src.qvisionframe.yolo_classical import (
    YOLOWithClassicalGate,
)
from src.qvisionframe.yolo_hybrid import (
    HybridPlacementConfig,
    YOLOWithPQFG,
)
from src.qvisionframe.yolo_linear import (
    YOLOWithLinearGate,
)
from src.qvisionframe.yolo_parameter_matched import (
    YOLOWithParameterMatchedGate,
)
from src.qvisionframe.yolo_trigonometric import (
    YOLOWithTrigonometricGate,
)


BaselineID = Literal[
    "B0",
    "B1",
    "B2",
    "B3",
    "B4",
    "B5",
    "B6",
    "B7",
    "B8",
]


@dataclass(frozen=True)
class BaselineBuildConfig:
    baseline_id: BaselineID
    architecture: str = "yolo11n.yaml"
    num_classes: int = 80
    target_layer: int = 10
    channels: int = 256
    hidden_dim: int = 64
    latent_dim: int = 4
    n_qubits: int = 4
    n_layers: int = 2
    alpha_init: float = 1.0e-3
    gamma: float = 0.5
    device: str = "cuda:0"


def _build_detector(
    architecture: str,
    num_classes: int,
    device: torch.device,
) -> nn.Module:
    if num_classes <= 0:
        raise ValueError("num_classes must be positive")

    detector = YOLO(architecture).model

    if detector.yaml.get("nc") != num_classes:
        detector.yaml["nc"] = num_classes

        # Rebuild through Ultralytics so the Detect head is created
        # with the dataset-specific number of classes.
        detector = YOLO(
            architecture,
            task="detect",
        ).model

        detector.yaml["nc"] = num_classes

        from ultralytics.nn.tasks import DetectionModel

        detector = DetectionModel(
            cfg=detector.yaml,
            ch=3,
            nc=num_classes,
            verbose=False,
        )

    return detector.to(device)


def build_baseline(
    config: BaselineBuildConfig,
) -> nn.Module:
    """
    Build one Q-VisionFrame baseline B0-B8.

    B0 returns the detector directly.
    B1-B8 return detector wrappers with an inserted gate.
    """

    device = torch.device(config.device)

    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            f"Requested {device}, but CUDA is unavailable"
        )

    detector = _build_detector(
        architecture=config.architecture,
        num_classes=config.num_classes,
        device=device,
    )

    if config.baseline_id == "B0":
        return detector

    placement = HybridPlacementConfig(
        target_layer=config.target_layer,
        channels=config.channels,
    )

    if config.baseline_id == "B1":
        model = YOLOWithLinearGate(
            yolo_network=detector,
            gate_config=LinearGateConfig(
                channels=config.channels,
                alpha_init=config.alpha_init,
            ),
            placement=placement,
        )

    elif config.baseline_id == "B2":
        model = YOLOWithChannelAttentionGate(
            yolo_network=detector,
            gate_config=ChannelAttentionGateConfig(
                channels=config.channels,
                reduction=16,
                alpha_init=config.alpha_init,
            ),
            placement=placement,
        )

    elif config.baseline_id == "B3":
        model = YOLOWithClassicalGate(
            yolo_network=detector,
            gate_config=ClassicalMLPGateConfig(
                channels=config.channels,
                latent_dim=config.latent_dim,
                hidden_dim=config.hidden_dim,
                transform_hidden_dim=8,
                alpha_init=config.alpha_init,
            ),
            placement=placement,
        )

    elif config.baseline_id == "B4":
        model = YOLOWithParameterMatchedGate(
            yolo_network=detector,
            gate_config=ParameterMatchedGateConfig(
                channels=config.channels,
                latent_dim=config.latent_dim,
                n_layers=config.n_layers,
                hidden_dim=config.hidden_dim,
                alpha_init=config.alpha_init,
            ),
            placement=placement,
        )

    elif config.baseline_id == "B5":
        model = YOLOWithTrigonometricGate(
            yolo_network=detector,
            gate_config=TrigonometricGateConfig(
                channels=config.channels,
                latent_dim=config.latent_dim,
                hidden_dim=config.hidden_dim,
                alpha_init=config.alpha_init,
            ),
            placement=placement,
        )

    elif config.baseline_id in {"B6", "B7", "B8"}:
        if config.baseline_id == "B6":
            entanglement = "ring"
            trainable_quantum = False
        elif config.baseline_id == "B7":
            entanglement = "none"
            trainable_quantum = True
        else:
            entanglement = "ring"
            trainable_quantum = True

        model = YOLOWithPQFG(
            yolo_network=detector,
            pqfg_config=PQFGConfig(
                channels=config.channels,
                n_qubits=config.n_qubits,
                n_layers=config.n_layers,
                hidden_dim=config.hidden_dim,
                alpha_init=config.alpha_init,
                entanglement=entanglement,
                gamma=config.gamma,
                trainable_quantum=trainable_quantum,
            ),
            placement=placement,
        )

    else:
        raise ValueError(
            f"Unsupported baseline ID: {config.baseline_id}"
        )

    return model.to(device)
