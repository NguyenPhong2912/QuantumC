from src.qvisionframe.classical_gate import (
    ParameterMatchedClassicalGate,
    ParameterMatchedGateConfig,
)
from src.qvisionframe.pqfg import PQFG, PQFGConfig


def parameter_count(module) -> int:
    return sum(
        parameter.numel()
        for parameter in module.parameters()
        if parameter.requires_grad
    )


def main() -> None:
    print("===== PARAMETER MATCHING CHECK =====")

    pqfg = PQFG(
        PQFGConfig(
            channels=256,
            n_qubits=4,
            n_layers=2,
            hidden_dim=64,
            alpha_init=1.0e-3,
            entanglement="ring",
        )
    )

    classical = ParameterMatchedClassicalGate(
        ParameterMatchedGateConfig(
            channels=256,
            latent_dim=4,
            n_layers=2,
            hidden_dim=64,
            alpha_init=1.0e-3,
        )
    )

    pqfg_total = parameter_count(pqfg)
    classical_total = parameter_count(classical)

    pqfg_transform = pqfg.quantum_weights.numel()

    classical_transform = sum(
        parameter.numel()
        for parameter in classical.latent_transform.parameters()
    )

    print("PQFG total parameters       :", pqfg_total)
    print("Classical total parameters  :", classical_total)
    print("PQFG transform parameters   :", pqfg_transform)
    print("Classical transform params  :", classical_transform)
    print(
        "Total parameter difference :",
        classical_total - pqfg_total,
    )

    assert pqfg_transform == 16
    assert classical_transform == 16
    assert classical_total == pqfg_total

    print("PARAMETER MATCHING          : PASSED")


if __name__ == "__main__":
    main()
