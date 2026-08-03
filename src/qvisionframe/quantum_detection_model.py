from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import torch
from torch import nn

from src.qvisionframe.gated_detection_model import (
    GatedDetectionModel,
)
from src.qvisionframe.pqfg import (
    PQFG,
    PQFGConfig,
)


QuantumBaselineID = Literal[
    "B6",
    "B7",
    "B8",
]


class QuantumGatedDetectionModel(
    GatedDetectionModel
):
    """
    Trainer-compatible YOLO DetectionModel with a PQFG
    feature gate registered at an internal detector layer.

    B6:
        ring entanglement, frozen quantum weights

    B7:
        no entanglement, trainable quantum weights

    B8:
        ring entanglement, trainable quantum weights
    """

    SUPPORTED_QUANTUM_BASELINES = {
        "B6",
        "B7",
        "B8",
    }

    def __init__(
        self,
        cfg: str | Path | dict[str, Any] = (
            "yolo11n.yaml"
        ),
        *,
        nc: int = 80,
        baseline_id: QuantumBaselineID = "B8",
        target_layer: int = 10,
        channels: int = 256,
        n_qubits: int = 4,
        n_layers: int = 2,
        hidden_dim: int | None = 64,
        alpha_init: float = 1.0e-3,
        gamma: float = 0.5,
        verbose: bool = True,
    ) -> None:
        if (
            baseline_id
            not in self.SUPPORTED_QUANTUM_BASELINES
        ):
            raise ValueError(
                "Unsupported quantum baseline "
                f"{baseline_id!r}; expected one of "
                f"{sorted(self.SUPPORTED_QUANTUM_BASELINES)}"
            )

        # Construct only the detector through the stable B0 path.
        # B0 does not register a feature hook.
        super().__init__(
            cfg=cfg,
            nc=nc,
            baseline_id="B0",
            target_layer=target_layer,
            channels=channels,
            latent_dim=n_qubits,
            hidden_dim=hidden_dim,
            n_layers=n_layers,
            alpha_init=alpha_init,
            verbose=verbose,
        )

        if baseline_id == "B6":
            entanglement = "ring"
            trainable_quantum = False
        elif baseline_id == "B7":
            entanglement = "none"
            trainable_quantum = True
        else:
            entanglement = "ring"
            trainable_quantum = True

        self.baseline_id = baseline_id

        self.qvf_gate_config = {
            "baseline_id": baseline_id,
            "target_layer": target_layer,
            "channels": channels,
            "n_qubits": n_qubits,
            "n_layers": n_layers,
            "hidden_dim": hidden_dim,
            "alpha_init": alpha_init,
            "entanglement": entanglement,
            "gamma": gamma,
            "trainable_quantum": (
                trainable_quantum
            ),
        }

        self.gate = PQFG(
            PQFGConfig(
                channels=channels,
                n_qubits=n_qubits,
                n_layers=n_layers,
                hidden_dim=hidden_dim,
                alpha_init=alpha_init,
                entanglement=entanglement,
                gamma=gamma,
                trainable_quantum=(
                    trainable_quantum
                ),
            )
        )

        self._gate_hook_handle = self.model[
            target_layer
        ].register_forward_hook(
            self._apply_feature_gate
        )

    @property
    def quantum_weights(self):
        return self.gate.quantum_weights

    def quantum_checkpoint_metadata(
        self,
    ) -> dict[str, Any]:
        """
        Return only JSON-compatible reconstruction metadata.
        No QNode, PennyLane device or simulator object is included.
        """
        return dict(self.qvf_gate_config)

    def _restore_quantum_fp64(self) -> None:
        """
        Preserve the numerical contract of the PQC parameters.

        Generic PyTorch/Ultralytics dtype conversions such as
        model.float() and model.half() recursively convert every
        floating parameter. PQFG quantum weights must remain FP64.
        """
        parameter = self.gate.quantum_weights

        if parameter.dtype != torch.float64:
            with torch.no_grad():
                parameter.data = parameter.data.to(
                    dtype=torch.float64
                )

            if parameter.grad is not None:
                parameter.grad.data = (
                    parameter.grad.data.to(
                        dtype=torch.float64
                    )
                )

    def float(self):
        """
        Convert the classical detector and gate path to FP32 while
        preserving gate.quantum_weights in FP64.
        """
        super().float()
        self._restore_quantum_fp64()
        return self

    def half(self):
        """
        Convert the classical path to FP16 while preserving the PQC
        parameters in FP64.

        Quantum checkpoints still must not use whole-object pickle;
        this method only prevents silent quantum precision loss.
        """
        super().half()
        self._restore_quantum_fp64()
        return self

    def close_gate_hook(self) -> None:
        super().close_gate_hook()
