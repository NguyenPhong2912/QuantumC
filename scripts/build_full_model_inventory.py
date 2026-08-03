from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
from pathlib import Path

from ultralytics import YOLO

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
from src.qvisionframe.pqfg import PQFG, PQFGConfig


@dataclass(frozen=True)
class FullModelRecord:
    baseline_id: str
    detector_parameters: int
    added_module_parameters: int
    total_model_parameters: int
    added_trainable_parameters: int
    total_trainable_parameters: int
    overhead_percent: float


def total_parameters(module) -> int:
    return sum(p.numel() for p in module.parameters())


def trainable_parameters(module) -> int:
    return sum(
        p.numel()
        for p in module.parameters()
        if p.requires_grad
    )


def make_record(
    baseline_id: str,
    detector_total: int,
    detector_trainable: int,
    module=None,
) -> FullModelRecord:
    added_total = 0 if module is None else total_parameters(module)
    added_trainable = (
        0 if module is None else trainable_parameters(module)
    )

    return FullModelRecord(
        baseline_id=baseline_id,
        detector_parameters=detector_total,
        added_module_parameters=added_total,
        total_model_parameters=detector_total + added_total,
        added_trainable_parameters=added_trainable,
        total_trainable_parameters=(
            detector_trainable + added_trainable
        ),
        overhead_percent=(
            100.0 * added_total / detector_total
        ),
    )


def main() -> None:
    print("===== FULL MODEL PARAMETER INVENTORY =====")

    detector = YOLO("yolo11n.yaml").model
    detector_total = total_parameters(detector)
    detector_trainable = trainable_parameters(detector)

    modules = {
        "B0": None,
        "B1": LinearClassicalGate(
            LinearGateConfig(
                channels=256,
                alpha_init=1.0e-3,
            )
        ),
        "B2": ChannelAttentionClassicalGate(
            ChannelAttentionGateConfig(
                channels=256,
                reduction=16,
                alpha_init=1.0e-3,
            )
        ),
        "B3": ClassicalMLPGate(
            ClassicalMLPGateConfig(
                channels=256,
                latent_dim=4,
                hidden_dim=64,
                transform_hidden_dim=8,
                alpha_init=1.0e-3,
            )
        ),
        "B4": ParameterMatchedClassicalGate(
            ParameterMatchedGateConfig(
                channels=256,
                latent_dim=4,
                n_layers=2,
                hidden_dim=64,
                alpha_init=1.0e-3,
            )
        ),
        "B5": TrigonometricClassicalGate(
            TrigonometricGateConfig(
                channels=256,
                latent_dim=4,
                hidden_dim=64,
                alpha_init=1.0e-3,
            )
        ),
        "B6": PQFG(
            PQFGConfig(
                channels=256,
                n_qubits=4,
                n_layers=2,
                hidden_dim=64,
                alpha_init=1.0e-3,
                entanglement="ring",
                gamma=0.5,
                trainable_quantum=False,
            )
        ),
        "B7": PQFG(
            PQFGConfig(
                channels=256,
                n_qubits=4,
                n_layers=2,
                hidden_dim=64,
                alpha_init=1.0e-3,
                entanglement="none",
                gamma=0.5,
                trainable_quantum=True,
            )
        ),
        "B8": PQFG(
            PQFGConfig(
                channels=256,
                n_qubits=4,
                n_layers=2,
                hidden_dim=64,
                alpha_init=1.0e-3,
                entanglement="ring",
                gamma=0.5,
                trainable_quantum=True,
            )
        ),
    }

    records = [
        make_record(
            baseline_id=baseline_id,
            detector_total=detector_total,
            detector_trainable=detector_trainable,
            module=module,
        )
        for baseline_id, module in modules.items()
    ]

    output_dir = Path("outputs/inventory")
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / "full_model_inventory.csv"

    with csv_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(asdict(records[0]).keys()),
        )
        writer.writeheader()

        for record in records:
            writer.writerow(asdict(record))

    print("Detector parameters :", detector_total)
    print("Detector trainable  :", detector_trainable)
    print("-" * 88)

    for record in records:
        print(
            f"{record.baseline_id:>2} | "
            f"added={record.added_module_parameters:>6} | "
            f"total={record.total_model_parameters:>8} | "
            f"trainable={record.total_trainable_parameters:>8} | "
            f"overhead={record.overhead_percent:.4f}%"
        )

    assert records[0].added_module_parameters == 0
    assert records[0].total_model_parameters == detector_total
    assert records[4].added_module_parameters == 34389
    assert records[8].added_module_parameters == 34389

    print("-" * 88)
    print("Inventory saved     :", csv_path)
    print("FULL MODEL INVENTORY: PASSED")


if __name__ == "__main__":
    main()
