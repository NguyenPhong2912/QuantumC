import torch

from src.qvisionframe.config import load_config
from src.qvisionframe.factory import build_hybrid_model


def main() -> None:
    print("===== CONFIG AND FACTORY CHECK =====")

    config = load_config(
        "configs/experiments/pqfg_p5_q4_l2_ring.yaml"
    )

    print("Experiment          :", config.experiment.name)
    print("Seed                :", config.experiment.seed)
    print("Architecture        :", config.model.architecture)
    print("Pretrained          :", config.model.pretrained)
    print("Target layer        :", config.placement.target_layer)
    print("Channels            :", config.placement.channels)
    print("Qubits              :", config.pqfg.n_qubits)
    print("Layers              :", config.pqfg.n_layers)
    print("Hidden dimension    :", config.pqfg.hidden_dim)
    print("Alpha init          :", config.pqfg.alpha_init)
    print("Entanglement        :", config.pqfg.entanglement)
    print("Gamma               :", config.pqfg.gamma)
    print("Detector device     :", config.runtime.detector_device)
    print("Quantum device      :", config.runtime.quantum_device)

    hybrid = build_hybrid_model(
        config=config,
        training=False,
    )

    print(
        "Projector device    :",
        hybrid.pqfg.projector[0].weight.device,
    )
    print(
        "Decoder device      :",
        hybrid.pqfg.decoder.weight.device,
    )
    print(
        "Quantum weight device:",
        hybrid.pqfg.quantum_weights.device,
    )

    assert config.experiment.name == "pqfg_p5_q4_l2_ring"
    assert config.placement.target_layer == 10
    assert config.placement.channels == 256
    assert config.pqfg.n_qubits == 4
    assert config.pqfg.n_layers == 2
    assert config.pqfg.hidden_dim == 64
    assert config.pqfg.entanglement == "ring"

    assert str(
        hybrid.pqfg.projector[0].weight.device
    ) == "cuda:0"

    assert str(
        hybrid.pqfg.decoder.weight.device
    ) == "cuda:0"

    assert str(
        hybrid.pqfg.quantum_weights.device
    ) == "cpu"

    hybrid.close()

    print("CONFIG AND FACTORY  : PASSED")


if __name__ == "__main__":
    main()
