from __future__ import annotations

from pathlib import Path
from typing import Any

from ultralytics.models.yolo.detect import (
    DetectionTrainer,
)
from ultralytics.utils import RANK

from src.qvisionframe.gated_detection_model import (
    GatedDetectionModel,
)


class QVisionDetectionTrainer(DetectionTrainer):
    """
    DetectionTrainer for classical Q-VisionFrame baselines B0-B5.
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
        *,
        baseline_id: str = "B0",
        target_layer: int = 10,
        gate_channels: int = 256,
        latent_dim: int = 4,
        hidden_dim: int | None = 64,
        transform_hidden_dim: int = 8,
        n_layers: int = 2,
        reduction: int = 16,
        alpha_init: float = 1.0e-3,
        overrides: dict[str, Any] | None = None,
        _callbacks: Any = None,
    ) -> None:
        if baseline_id not in self.SUPPORTED_BASELINES:
            raise ValueError(
                "QVisionDetectionTrainer currently "
                "supports B0-B5, received "
                f"{baseline_id}"
            )

        self.qvf_baseline_id = baseline_id
        self.qvf_target_layer = target_layer
        self.qvf_gate_channels = gate_channels
        self.qvf_latent_dim = latent_dim
        self.qvf_hidden_dim = hidden_dim
        self.qvf_transform_hidden_dim = (
            transform_hidden_dim
        )
        self.qvf_n_layers = n_layers
        self.qvf_reduction = reduction
        self.qvf_alpha_init = alpha_init

        kwargs: dict[str, Any] = {
            "overrides": overrides or {},
        }

        if _callbacks is not None:
            kwargs["_callbacks"] = _callbacks

        super().__init__(**kwargs)

    def get_model(
        self,
        cfg: str | Path | dict[str, Any] | None = None,
        weights: Any = None,
        verbose: bool = True,
    ) -> GatedDetectionModel:
        if cfg is None:
            cfg = self.args.model

        model = GatedDetectionModel(
            cfg=cfg,
            nc=self.data["nc"],
            baseline_id=self.qvf_baseline_id,
            target_layer=self.qvf_target_layer,
            channels=self.qvf_gate_channels,
            latent_dim=self.qvf_latent_dim,
            hidden_dim=self.qvf_hidden_dim,
            transform_hidden_dim=(
                self.qvf_transform_hidden_dim
            ),
            n_layers=self.qvf_n_layers,
            reduction=self.qvf_reduction,
            alpha_init=self.qvf_alpha_init,
            verbose=verbose and RANK == -1,
        )

        if weights:
            model.load(weights)

        return model
