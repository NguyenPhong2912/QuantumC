from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
from pathlib import Path

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
class BaselineRecord:
    baseline_id: str
    name: str
    transform: str
    placement: str
    target_layer: int | None
    channels: int | None
    total_parameters: int
    trainable_parameters: int
    quantum_parameters: int
    quantum_trainable: bool | None
    status: str
    notes: str


def total_parameters(module) -> int:
    return sum(
        parameter.numel()
        for parameter in module.parameters()
    )


def trainable_parameters(module) -> int:
    return sum(
        parameter.numel()
        for parameter in module.parameters()
        if parameter.requires_grad
    )


def main() -> None:
    print("===== Q-VISIONFRAME BASELINE INVENTORY =====")

    linear_gate = LinearClassicalGate(
        LinearGateConfig(
            channels=256,
            alpha_init=1.0e-3,
        )
    )

    channel_attention = ChannelAttentionClassicalGate(
        ChannelAttentionGateConfig(
            channels=256,
            reduction=16,
            alpha_init=1.0e-3,
        )
    )

    classical_mlp = ClassicalMLPGate(
        ClassicalMLPGateConfig(
            channels=256,
            latent_dim=4,
            hidden_dim=64,
            transform_hidden_dim=8,
            alpha_init=1.0e-3,
        )
    )

    parameter_matched = ParameterMatchedClassicalGate(
        ParameterMatchedGateConfig(
            channels=256,
            latent_dim=4,
            n_layers=2,
            hidden_dim=64,
            alpha_init=1.0e-3,
        )
    )

    trigonometric = TrigonometricClassicalGate(
        TrigonometricGateConfig(
            channels=256,
            latent_dim=4,
            hidden_dim=64,
            alpha_init=1.0e-3,
        )
    )

    frozen_pqfg = PQFG(
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
    )

    trainable_none = PQFG(
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
    )

    trainable_linear = PQFG(
        PQFGConfig(
            channels=256,
            n_qubits=4,
            n_layers=2,
            hidden_dim=64,
            alpha_init=1.0e-3,
            entanglement="linear",
            gamma=0.5,
            trainable_quantum=True,
        )
    )

    trainable_ring = PQFG(
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
    )

    records = [
        BaselineRecord(
            baseline_id="B0",
            name="YOLO11n",
            transform="none",
            placement="none",
            target_layer=None,
            channels=None,
            total_parameters=0,
            trainable_parameters=0,
            quantum_parameters=0,
            quantum_trainable=None,
            status="architecture pending benchmark",
            notes="Detector-only reference; gate parameters excluded.",
        ),
        BaselineRecord(
            baseline_id="B1",
            name="YOLO11n + GAP/GMP Linear Gate",
            transform="linear classical",
            placement="after backbone P5",
            target_layer=10,
            channels=256,
            total_parameters=total_parameters(linear_gate),
            trainable_parameters=trainable_parameters(linear_gate),
            quantum_parameters=0,
            quantum_trainable=None,
            status="implemented and tested",
            notes="Direct 512-to-256 linear channel gate.",
        ),
        BaselineRecord(
            baseline_id="B2",
            name="YOLO11n + SE/CBAM-style Gate",
            transform="attention classical",
            placement="after backbone P5",
            target_layer=10,
            channels=256,
            total_parameters=total_parameters(channel_attention),
            trainable_parameters=trainable_parameters(
                channel_attention
            ),
            quantum_parameters=0,
            quantum_trainable=None,
            status="implemented and tested",
            notes=(
                "Shared reduction-16 MLP over GAP and GMP descriptors."
            ),
        ),
        BaselineRecord(
            baseline_id="B3",
            name="YOLO11n + Classical MLP Gate",
            transform="MLP 4-8-4",
            placement="after backbone P5",
            target_layer=10,
            channels=256,
            total_parameters=total_parameters(classical_mlp),
            trainable_parameters=trainable_parameters(classical_mlp),
            quantum_parameters=0,
            quantum_trainable=None,
            status="implemented and tested",
            notes="Same interface as PQFG; not exactly parameter matched.",
        ),
        BaselineRecord(
            baseline_id="B4",
            name="YOLO11n + Parameter-Matched Classical Gate",
            transform="2 elementwise affine layers",
            placement="after backbone P5",
            target_layer=10,
            channels=256,
            total_parameters=total_parameters(parameter_matched),
            trainable_parameters=trainable_parameters(parameter_matched),
            quantum_parameters=0,
            quantum_trainable=None,
            status="implemented and tested",
            notes="Exactly matched to PQFG total and transform parameters.",
        ),
        BaselineRecord(
            baseline_id="B5",
            name="YOLO11n + Trigonometric Gate",
            transform="sin/cos classical",
            placement="after backbone P5",
            target_layer=10,
            channels=256,
            total_parameters=total_parameters(trigonometric),
            trainable_parameters=trainable_parameters(trigonometric),
            quantum_parameters=0,
            quantum_trainable=None,
            status="implemented and tested",
            notes="Periodic nonlinear classical control.",
        ),
        BaselineRecord(
            baseline_id="B6",
            name="YOLO11n + Frozen Random PQC",
            transform="4-qubit 2-layer ring PQC",
            placement="after backbone P5",
            target_layer=10,
            channels=256,
            total_parameters=total_parameters(frozen_pqfg),
            trainable_parameters=trainable_parameters(frozen_pqfg),
            quantum_parameters=frozen_pqfg.quantum_weights.numel(),
            quantum_trainable=False,
            status="implemented and tested",
            notes="Projector, decoder and alpha trainable; PQC frozen.",
        ),
        BaselineRecord(
            baseline_id="B7",
            name="YOLO11n + Trainable PQC No Entanglement",
            transform="4-qubit 2-layer separable PQC",
            placement="after backbone P5",
            target_layer=10,
            channels=256,
            total_parameters=total_parameters(trainable_none),
            trainable_parameters=trainable_parameters(trainable_none),
            quantum_parameters=trainable_none.quantum_weights.numel(),
            quantum_trainable=True,
            status="implemented and tested",
            notes="Trainable circuit without entangling gates.",
        ),
        BaselineRecord(
            baseline_id="B8",
            name="YOLO11n + Trainable PQC Ring Entanglement",
            transform="4-qubit 2-layer ring PQC",
            placement="after backbone P5",
            target_layer=10,
            channels=256,
            total_parameters=total_parameters(trainable_ring),
            trainable_parameters=trainable_parameters(trainable_ring),
            quantum_parameters=trainable_ring.quantum_weights.numel(),
            quantum_trainable=True,
            status="implemented and tested",
            notes=(
                "Primary PQFG candidate. Linear topology retained "
                "as an internal topology ablation."
            ),
        ),
    ]

    output_dir = Path("outputs/inventory")
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / "baseline_inventory.csv"

    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(asdict(records[0]).keys()),
        )
        writer.writeheader()
        for record in records:
            writer.writerow(asdict(record))

    for record in records:
        print(
            f"{record.baseline_id:>2} | "
            f"{record.total_parameters:>6} total | "
            f"{record.trainable_parameters:>6} trainable | "
            f"{record.transform}"
        )

    print("-" * 72)
    print(
        "Linear PQC topology parameters:",
        total_parameters(trainable_linear),
    )
    print("Inventory saved:", csv_path)

    assert total_parameters(linear_gate) == 131329
    assert total_parameters(channel_attention) == 8465
    assert total_parameters(parameter_matched) == 34389
    assert total_parameters(trainable_ring) == 34389
    assert trainable_parameters(frozen_pqfg) == 34373
    assert frozen_pqfg.quantum_weights.numel() == 16
    assert not frozen_pqfg.quantum_weights.requires_grad
    assert trainable_ring.quantum_weights.requires_grad

    print("BASELINE INVENTORY   : PASSED")


if __name__ == "__main__":
    main()
