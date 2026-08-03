from pathlib import Path

from src.qvisionframe.config import load_config
from src.qvisionframe.factory import build_hybrid_model


CONFIGS = {
    "none": Path(
        "configs/experiments/pqfg_p5_q4_l2_none.yaml"
    ),
    "linear": Path(
        "configs/experiments/pqfg_p5_q4_l2_linear.yaml"
    ),
    "ring": Path(
        "configs/experiments/pqfg_p5_q4_l2_ring.yaml"
    ),
}


def main() -> None:
    print("===== ENTANGLEMENT CONFIG CHECK =====")

    for expected_entanglement, path in CONFIGS.items():
        config = load_config(path)

        assert config.pqfg.entanglement == expected_entanglement
        assert config.pqfg.n_qubits == 4
        assert config.pqfg.n_layers == 2
        assert config.placement.target_layer == 10
        assert config.placement.channels == 256

        hybrid = build_hybrid_model(
            config=config,
            training=False,
        )

        print("Config              :", path.name)
        print("Experiment          :", config.experiment.name)
        print("Entanglement        :", config.pqfg.entanglement)
        print("Qubits / layers     :", (
            config.pqfg.n_qubits,
            config.pqfg.n_layers,
        ))
        print("Projector device    :", (
            hybrid.pqfg.projector[0].weight.device
        ))
        print("Quantum device      :", (
            hybrid.pqfg.quantum_weights.device
        ))
        print("-" * 60)

        assert str(
            hybrid.pqfg.projector[0].weight.device
        ) == "cuda:0"

        assert str(
            hybrid.pqfg.quantum_weights.device
        ) == "cpu"

        hybrid.close()

    print("ENTANGLEMENT CONFIGS: PASSED")


if __name__ == "__main__":
    main()
