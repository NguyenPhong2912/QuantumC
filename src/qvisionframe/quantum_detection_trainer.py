from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

import torch
from ultralytics import __version__
from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.utils import LOGGER, RANK

from src.qvisionframe.quantum_checkpoint import (
    build_quantum_checkpoint,
    load_quantum_checkpoint,
    reconstruct_quantum_model,
    save_quantum_checkpoint,
)
from src.qvisionframe.quantum_detection_model import (
    QuantumBaselineID,
    QuantumGatedDetectionModel,
)


class QuantumDetectionTrainer(DetectionTrainer):
    """
    Ultralytics DetectionTrainer for Q-VisionFrame B6-B8.

    The training loop, optimizer, scheduler, EMA and training-time
    validator remain inherited from DetectionTrainer.

    Checkpoint serialization is replaced because PQFG contains a
    PennyLane Lightning runtime object after its first forward and
    quantum_weights must remain FP64.
    """

    def __init__(
        self,
        *args,
        quantum_baseline_id: QuantumBaselineID = "B8",
        quantum_target_layer: int = 10,
        quantum_channels: int = 256,
        quantum_n_qubits: int = 4,
        quantum_n_layers: int = 2,
        quantum_hidden_dim: int | None = 64,
        quantum_alpha_init: float = 1.0e-3,
        quantum_gamma: float = 0.5,
        quantum_resume_path: str | Path | None = None,
        **kwargs,
    ) -> None:
        self.quantum_resume_path = (
            None
            if quantum_resume_path is None
            else Path(quantum_resume_path).expanduser().resolve()
        )
        self.qvf_resume_checkpoint = None

        if quantum_baseline_id not in {
            "B6",
            "B7",
            "B8",
        }:
            raise ValueError(
                "quantum_baseline_id must be one of "
                "{'B6', 'B7', 'B8'}"
            )

        self.quantum_baseline_id = (
            quantum_baseline_id
        )
        self.quantum_target_layer = int(
            quantum_target_layer
        )
        self.quantum_channels = int(
            quantum_channels
        )
        self.quantum_n_qubits = int(
            quantum_n_qubits
        )
        self.quantum_n_layers = int(
            quantum_n_layers
        )
        self.quantum_hidden_dim = (
            quantum_hidden_dim
        )
        self.quantum_alpha_init = float(
            quantum_alpha_init
        )
        self.quantum_gamma = float(
            quantum_gamma
        )

        super().__init__(*args, **kwargs)

    def check_resume(
        self,
        overrides: dict[str, Any],
    ) -> None:
        """
        Enable explicit Q-VisionFrame tensor-only resume.

        Standard Ultralytics resume expects a checkpoint containing a
        pickled model object. Q-VisionFrame resumes from reconstruction
        metadata plus state dictionaries instead.
        """
        if self.quantum_resume_path is None:
            super().check_resume(overrides)
            return

        if not self.quantum_resume_path.exists():
            raise FileNotFoundError(
                "Quantum resume checkpoint not found: "
                f"{self.quantum_resume_path}"
            )

        payload = load_quantum_checkpoint(
            self.quantum_resume_path,
            map_location="cpu",
        )

        checkpoint_baseline = payload[
            "model_metadata"
        ]["gate"]["baseline_id"]

        if checkpoint_baseline != self.quantum_baseline_id:
            raise ValueError(
                "Quantum resume baseline mismatch: "
                f"trainer={self.quantum_baseline_id}, "
                f"checkpoint={checkpoint_baseline}"
            )

        self.qvf_resume_checkpoint = payload
        self.resume = True

        # Keep current overrides as the source of run configuration.
        # The checkpoint supplies model/training state, while epochs,
        # data, device and output directory remain explicitly defined
        # by the new trainer invocation.
        self.args.resume = str(
            self.quantum_resume_path
        )

        LOGGER.info(
            "QVF tensor-only resume enabled: "
            f"{self.quantum_resume_path}"
        )

    def setup_model(
        self,
    ) -> dict[str, Any] | None:
        """
        Reconstruct a quantum model from tensor-only checkpoint state.

        For non-resume runs, retain the ordinary DetectionTrainer
        setup path.
        """
        if self.quantum_resume_path is None:
            return super().setup_model()

        payload = self.qvf_resume_checkpoint

        if payload is None:
            payload = load_quantum_checkpoint(
                self.quantum_resume_path,
                map_location="cpu",
            )
            self.qvf_resume_checkpoint = payload

        metadata = payload["model_metadata"]
        gate = metadata["gate"]

        # Synchronize constructor metadata with the checkpoint.
        self.quantum_baseline_id = gate["baseline_id"]
        self.quantum_target_layer = int(
            gate["target_layer"]
        )
        self.quantum_channels = int(
            gate["channels"]
        )
        self.quantum_n_qubits = int(
            gate["n_qubits"]
        )
        self.quantum_n_layers = int(
            gate["n_layers"]
        )
        self.quantum_hidden_dim = gate[
            "hidden_dim"
        ]
        self.quantum_alpha_init = float(
            gate["alpha_init"]
        )
        self.quantum_gamma = float(
            gate["gamma"]
        )

        self.model = reconstruct_quantum_model(
            payload,
            state_key="model_state_dict",
            device="cpu",
            strict=True,
            verbose=RANK in {-1, 0},
        )

        LOGGER.info(
            "QVF reconstructed raw model from "
            f"epoch {payload['epoch']} checkpoint"
        )

        return payload

    def get_model(
        self,
        cfg: str | Path | dict[str, Any] | None = None,
        weights: torch.nn.Module | None = None,
        verbose: bool = True,
    ) -> QuantumGatedDetectionModel:
        """
        Construct a trainer-compatible B6-B8 detection model.
        """
        model = QuantumGatedDetectionModel(
            cfg=cfg or "yolo11n.yaml",
            nc=int(self.data["nc"]),
            baseline_id=self.quantum_baseline_id,
            target_layer=self.quantum_target_layer,
            channels=self.quantum_channels,
            n_qubits=self.quantum_n_qubits,
            n_layers=self.quantum_n_layers,
            hidden_dim=self.quantum_hidden_dim,
            alpha_init=self.quantum_alpha_init,
            gamma=self.quantum_gamma,
            verbose=verbose,
        )

        if weights is not None:
            model.load(weights)

        return model

    def build_optimizer(
        self,
        model,
        name="auto",
        lr=0.001,
        momentum=0.9,
        decay=1.0e-5,
        iterations=1.0e5,
    ):
        """
        Build the Ultralytics optimizer while enforcing the B6
        frozen-quantum contract.

        Ultralytics re-enables floating parameters that are frozen
        outside its standard freeze list. Its default optimizer builder
        also includes parameters regardless of requires_grad. Therefore
        B6 quantum_weights must be restored to requires_grad=False and
        removed explicitly from every optimizer parameter group.
        """
        raw_model = model.module if hasattr(
            model,
            "module",
        ) else model

        if not isinstance(
            raw_model,
            QuantumGatedDetectionModel,
        ):
            raise TypeError(
                "Expected QuantumGatedDetectionModel, got "
                f"{type(raw_model).__name__}"
            )

        quantum_parameter = (
            raw_model.gate.quantum_weights
        )

        # Snapshot before the first optimizer step. This is used by
        # lifecycle smoke tests to prove that B6 remains frozen while
        # B7/B8 quantum parameters receive real updates.
        self.qvf_initial_quantum_weights = (
            quantum_parameter.detach()
            .cpu()
            .clone()
        )

        if self.quantum_baseline_id == "B6":
            quantum_parameter.requires_grad_(
                False
            )

        optimizer = super().build_optimizer(
            model=model,
            name=name,
            lr=lr,
            momentum=momentum,
            decay=decay,
            iterations=iterations,
        )

        removed_parameters = 0

        if self.quantum_baseline_id == "B6":
            for group in optimizer.param_groups:
                original_parameters = list(
                    group["params"]
                )

                filtered_parameters = [
                    parameter
                    for parameter
                    in original_parameters
                    if parameter is not quantum_parameter
                ]

                removed_parameters += (
                    len(original_parameters)
                    - len(filtered_parameters)
                )

                group["params"] = (
                    filtered_parameters
                )

            # Defensive cleanup in case a future optimizer initializes
            # state eagerly during construction.
            optimizer.state.pop(
                quantum_parameter,
                None,
            )

            quantum_in_optimizer = any(
                parameter is quantum_parameter
                for group in optimizer.param_groups
                for parameter in group["params"]
            )

            if quantum_parameter.requires_grad:
                raise RuntimeError(
                    "B6 quantum_weights were re-enabled"
                )

            if quantum_in_optimizer:
                raise RuntimeError(
                    "B6 quantum_weights remain in optimizer"
                )

            if removed_parameters != 1:
                raise RuntimeError(
                    "Expected to remove exactly one B6 quantum "
                    "parameter tensor, removed "
                    f"{removed_parameters}"
                )

            LOGGER.info(
                "QVF B6 optimizer contract: "
                "gate.quantum_weights frozen and excluded "
                "from optimizer"
            )

        else:
            if not quantum_parameter.requires_grad:
                raise RuntimeError(
                    f"{self.quantum_baseline_id} quantum_weights "
                    "must remain trainable"
                )

            quantum_in_optimizer = any(
                parameter is quantum_parameter
                for group in optimizer.param_groups
                for parameter in group["params"]
            )

            if not quantum_in_optimizer:
                raise RuntimeError(
                    f"{self.quantum_baseline_id} quantum_weights "
                    "missing from optimizer"
                )

        self.qvf_optimizer_audit = {
            "baseline_id": (
                self.quantum_baseline_id
            ),
            "quantum_requires_grad": bool(
                quantum_parameter.requires_grad
            ),
            "quantum_in_optimizer": bool(
                quantum_in_optimizer
            ),
            "removed_parameter_tensors": int(
                removed_parameters
            ),
        }

        LOGGER.info(
            "QVF quantum optimizer contract verified: "
            f"{self.qvf_optimizer_audit}"
        )

        return optimizer

    def _quantum_train_args(
        self,
    ) -> dict[str, Any]:
        """
        Add reconstruction information to ordinary trainer arguments.
        """
        values = dict(vars(self.args))

        values.update(
            {
                "qvf_checkpoint_format": (
                    "qvisionframe.quantum.v1"
                ),
                "qvf_quantum_baseline_id": (
                    self.quantum_baseline_id
                ),
                "qvf_quantum_target_layer": (
                    self.quantum_target_layer
                ),
                "qvf_quantum_channels": (
                    self.quantum_channels
                ),
                "qvf_quantum_n_qubits": (
                    self.quantum_n_qubits
                ),
                "qvf_quantum_n_layers": (
                    self.quantum_n_layers
                ),
                "qvf_quantum_hidden_dim": (
                    self.quantum_hidden_dim
                ),
                "qvf_quantum_alpha_init": (
                    self.quantum_alpha_init
                ),
                "qvf_quantum_gamma": (
                    self.quantum_gamma
                ),
            }
        )

        return values

    def save_model(self) -> bool:
        """
        Save a tensor-only quantum checkpoint.

        This method intentionally does not:
        - deepcopy either model;
        - pickle an nn.Module;
        - call model.half();
        - call Ultralytics strip_optimizer().
        """
        if self.ema is None:
            raise RuntimeError(
                "Quantum checkpoint requires EMA"
            )

        raw_model = self.model

        if hasattr(raw_model, "module"):
            raw_model = raw_model.module

        ema_model = self.ema.ema

        if not isinstance(
            raw_model,
            QuantumGatedDetectionModel,
        ):
            raise TypeError(
                "Raw model is not "
                "QuantumGatedDetectionModel: "
                f"{type(raw_model).__name__}"
            )

        if not isinstance(
            ema_model,
            QuantumGatedDetectionModel,
        ):
            raise TypeError(
                "EMA model is not "
                "QuantumGatedDetectionModel: "
                f"{type(ema_model).__name__}"
            )

        train_metrics = {
            **dict(self.metrics or {}),
            "fitness": (
                None
                if self.fitness is None
                else float(self.fitness)
            ),
        }

        payload = build_quantum_checkpoint(
            model=raw_model,
            ema_model=ema_model,
            ema_updates=self.ema.updates,
            optimizer=self.optimizer,
            scaler_state_dict=(
                None
                if self.scaler is None
                else self.scaler.state_dict()
            ),
            epoch=self.epoch,
            best_fitness=self.best_fitness,
            train_args=self._quantum_train_args(),
            train_metrics=train_metrics,
            train_results=self.read_results_csv(),
            extra_metadata={
                "date": datetime.now().isoformat(),
                "ultralytics_version": __version__,
                "trainer_class": (
                    type(self).__name__
                ),
                "checkpoint_role": "training",
            },
        )

        save_quantum_checkpoint(
            payload,
            self.last,
        )

        if self.best_fitness == self.fitness:
            save_quantum_checkpoint(
                payload,
                self.best,
            )

        if (
            self.save_period > 0
            and self.epoch % self.save_period == 0
        ):
            save_quantum_checkpoint(
                payload,
                self.wdir
                / f"epoch{self.epoch}.pt",
            )

        return True

    def _load_checkpoint_state(
        self,
        checkpoint: dict[str, Any],
    ) -> None:
        """
        Restore optimizer, scaler, EMA and fitness from a tensor-only
        Q-VisionFrame checkpoint.
        """
        if (
            checkpoint.get("qvf_format")
            != "qvisionframe.quantum.v1"
        ):
            super()._load_checkpoint_state(
                checkpoint
            )
            return

        optimizer_state = checkpoint.get(
            "optimizer_state_dict"
        )

        if optimizer_state is not None:
            self.optimizer.load_state_dict(
                optimizer_state
            )

        scaler_state = checkpoint.get(
            "scaler_state_dict"
        )

        if (
            scaler_state is not None
            and self.scaler is not None
        ):
            self.scaler.load_state_dict(
                scaler_state
            )

        if self.ema is None:
            raise RuntimeError(
                "EMA container must exist before quantum resume"
            )

        load_result = self.ema.ema.load_state_dict(
            checkpoint["ema_state_dict"],
            strict=True,
        )

        if load_result.missing_keys:
            raise RuntimeError(
                "Missing EMA state keys during resume: "
                f"{load_result.missing_keys}"
            )

        if load_result.unexpected_keys:
            raise RuntimeError(
                "Unexpected EMA state keys during resume: "
                f"{load_result.unexpected_keys}"
            )

        self.ema.ema._restore_quantum_fp64()
        self.ema.updates = int(
            checkpoint["ema_updates"]
        )

        self.best_fitness = checkpoint.get(
            "best_fitness"
        )

        optimizer_state_devices = set()
        optimizer_state_dtypes = set()
        optimizer_state_tensor_count = 0

        for state in self.optimizer.state.values():
            for value in state.values():
                if isinstance(value, torch.Tensor):
                    optimizer_state_tensor_count += 1
                    optimizer_state_devices.add(
                        str(value.device)
                    )
                    optimizer_state_dtypes.add(
                        str(value.dtype)
                    )

        self.qvf_resume_audit = {
            "checkpoint_epoch": int(
                checkpoint["epoch"]
            ),
            "ema_updates": int(
                self.ema.updates
            ),
            "optimizer_state_entries": (
                0
                if optimizer_state is None
                else len(
                    optimizer_state["state"]
                )
            ),
            "raw_quantum_dtype": str(
                self.model.quantum_weights.dtype
            ),
            "ema_quantum_dtype": str(
                self.ema.ema.quantum_weights.dtype
            ),
            "optimizer_state_tensor_count": int(
                optimizer_state_tensor_count
            ),
            "optimizer_state_devices": sorted(
                optimizer_state_devices
            ),
            "optimizer_state_dtypes": sorted(
                optimizer_state_dtypes
            ),
        }

        LOGGER.info(
            "QVF resume state restored: "
            f"epoch={checkpoint['epoch']}, "
            f"ema_updates={self.ema.updates}, "
            "optimizer_entries="
            f"{self.qvf_resume_audit['optimizer_state_entries']}"
        )

    def _load_final_quantum_model(
        self,
        path: str | Path,
    ) -> QuantumGatedDetectionModel:
        payload = load_quantum_checkpoint(
            path,
            map_location="cpu",
        )

        model = reconstruct_quantum_model(
            payload,
            state_key="ema_state_dict",
            device=self.device,
            strict=True,
            verbose=False,
        )

        model.args = self.args
        model.names = self.model.names
        model.stride = self.model.stride
        model.eval()
        model.float()

        return model

    def final_eval(self) -> None:
        """
        Run final validation without strip_optimizer() and without
        loading the checkpoint through Ultralytics AutoBackend.

        The reconstructed EMA model is installed into the existing EMA
        container, allowing the validator's training branch to reuse
        the already-built validation dataloader and calculate val loss.
        """
        checkpoint_path: Path | None = None

        if self.best.exists():
            checkpoint_path = self.best
        elif self.last.exists():
            checkpoint_path = self.last

        if checkpoint_path is None:
            LOGGER.warning(
                "No quantum checkpoint found for final validation"
            )
            return

        LOGGER.info(
            f"\nValidating quantum checkpoint "
            f"{checkpoint_path}..."
        )

        final_model = (
            self._load_final_quantum_model(
                checkpoint_path
            )
        )

        previous_ema_model = self.ema.ema
        self.ema.ema = final_model

        try:
            self.validator.args.plots = (
                self.args.plots
            )
            self.validator.args.compile = False

            metrics = self.validator(self)

            if metrics is not None:
                self.metrics = dict(metrics)
                self.metrics.pop("fitness", None)

            self.epoch += 1
            self.run_callbacks(
                "on_fit_epoch_end"
            )
            self.epoch -= 1

        finally:
            self.ema.ema = previous_ema_model

            if hasattr(
                final_model,
                "close_gate_hook",
            ):
                final_model.close_gate_hook()
