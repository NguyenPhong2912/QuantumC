from __future__ import annotations

import torch
from ultralytics import YOLO
from ultralytics.cfg import get_cfg
from ultralytics.utils import DEFAULT_CFG

from src.qvisionframe.config import QVisionFrameConfig
from src.qvisionframe.yolo_hybrid import YOLOWithPQFG


def build_hybrid_model(
    config: QVisionFrameConfig,
    training: bool = True,
) -> YOLOWithPQFG:
    """Build YOLO with PQFG from a validated experiment config."""

    if config.model.pretrained:
        source = f"{config.model.detector}.pt"
    else:
        source = config.model.architecture

    yolo_wrapper = YOLO(source)
    network = yolo_wrapper.model

    detector_device = torch.device(
        config.runtime.detector_device
    )
    network = network.to(detector_device)

    if training:
        network.train()
        network.args = get_cfg(
            cfg=DEFAULT_CFG,
            overrides=yolo_wrapper.overrides,
        )
        network.criterion = None
    else:
        network.eval()

    hybrid = YOLOWithPQFG(
        yolo_network=network,
        pqfg_config=config.pqfg,
        placement=config.placement,
    )

    # Classical PQFG components run with YOLO on CUDA.
    hybrid.pqfg.projector.to(detector_device)
    hybrid.pqfg.decoder.to(detector_device)
    hybrid.pqfg.alpha.data = (
        hybrid.pqfg.alpha.data.to(detector_device)
    )

    # lightning.qubit executes on CPU.
    hybrid.pqfg.quantum_weights.data = (
        hybrid.pqfg.quantum_weights.data.cpu()
    )

    return hybrid
