from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Any

import torch

from src.qvisionframe.pqfg import (
    PQFG,
    PQFGConfig,
)


BASELINES: dict[str, dict[str, Any]] = {
    "B6": {
        "entanglement": "ring",
        "trainable_quantum": False,
    },
    "B7": {
        "entanglement": "none",
        "trainable_quantum": True,
    },
    "B8": {
        "entanglement": "ring",
        "trainable_quantum": True,
    },
}


def make_config(
    baseline_id: str,
) -> PQFGConfig:
    values = BASELINES[baseline_id]

    return PQFGConfig(
        channels=256,
        n_qubits=4,
        n_layers=2,
        hidden_dim=64,
        alpha_init=1.0e-3,
        entanglement=values["entanglement"],
        gamma=0.5,
        trainable_quantum=values[
            "trainable_quantum"
        ],
    )


def clone_state(
    module: torch.nn.Module,
) -> dict[str, torch.Tensor]:
    return {
        key: tensor.detach().cpu().clone()
        for key, tensor
        in module.state_dict().items()
    }


def compare_states(
    first: dict[str, torch.Tensor],
    second: dict[str, torch.Tensor],
) -> dict[str, Any]:
    if first.keys() != second.keys():
        raise RuntimeError(
            "State dictionary keys differ"
        )

    exact = 0
    maximum = 0.0
    nonexact: list[str] = []

    for key in first:
        tensor_01 = first[key]
        tensor_02 = second[key]

        if torch.equal(tensor_01, tensor_02):
            exact += 1
        else:
            nonexact.append(key)

        difference = (
            tensor_01.float()
            - tensor_02.float()
        )

        if difference.numel():
            maximum = max(
                maximum,
                float(difference.abs().max()),
            )

    return {
        "exact": exact,
        "total": len(first),
        "max_abs_difference": maximum,
        "nonexact_keys": nonexact,
    }


def gradient_report(
    gate: PQFG,
    input_tensor: torch.Tensor,
) -> dict[str, dict[str, Any]]:
    gate.zero_grad(set_to_none=True)

    output = gate(input_tensor)
    loss = output.square().mean()
    loss.backward()

    return {
        name: {
            "requires_grad": parameter.requires_grad,
            "gradient_present": (
                parameter.grad is not None
            ),
            "gradient_finite": (
                bool(
                    torch.isfinite(
                        parameter.grad
                    ).all()
                )
                if parameter.grad is not None
                else None
            ),
            "gradient_norm": (
                float(
                    parameter.grad.detach()
                    .float()
                    .norm()
                )
                if parameter.grad is not None
                else None
            ),
        }
        for name, parameter
        in gate.named_parameters()
    }


def main() -> None:
    print(
        "===== PQFG STATE_DICT ROUNDTRIP ====="
    )

    root = Path(__file__).resolve().parents[1]

    report: dict[str, Any] = {}

    for baseline_id in BASELINES:
        print()
        print(f"===== {baseline_id} =====")

        torch.manual_seed(42)

        config = make_config(baseline_id)
        source = PQFG(config).eval()

        input_tensor = torch.randn(
            2,
            256,
            4,
            4,
            dtype=torch.float32,
        )

        with torch.no_grad():
            source_output = source(
                input_tensor
            ).detach().cpu()

        source_state = clone_state(source)

        # Save only tensors and plain configuration values.
        payload = {
            "format_version": 1,
            "baseline_id": baseline_id,
            "config": {
                "channels": config.channels,
                "n_qubits": config.n_qubits,
                "n_layers": config.n_layers,
                "hidden_dim": config.hidden_dim,
                "alpha_init": config.alpha_init,
                "entanglement": (
                    config.entanglement
                ),
                "gamma": config.gamma,
                "trainable_quantum": (
                    config.trainable_quantum
                ),
            },
            "state_dict": source_state,
        }

        memory = io.BytesIO()

        torch.save(
            payload,
            memory,
        )

        serialized_bytes = memory.tell()
        memory.seek(0)

        loaded_payload = torch.load(
            memory,
            map_location="cpu",
            weights_only=True,
        )

        restored_config = PQFGConfig(
            **loaded_payload["config"]
        )

        # This reconstructs a fresh PennyLane device,
        # QNode and Lightning state vector.
        restored = PQFG(
            restored_config
        ).eval()

        load_result = restored.load_state_dict(
            loaded_payload["state_dict"],
            strict=True,
        )

        with torch.no_grad():
            restored_output = restored(
                input_tensor
            ).detach().cpu()

        restored_state = clone_state(restored)

        state_comparison = compare_states(
            source_state,
            restored_state,
        )

        output_difference = (
            source_output.float()
            - restored_output.float()
        )

        output_exact = torch.equal(
            source_output,
            restored_output,
        )

        output_max_difference = float(
            output_difference.abs().max()
        )

        torch.manual_seed(123)

        gradient_input = torch.randn(
            1,
            256,
            4,
            4,
            dtype=torch.float32,
        )

        gradients = gradient_report(
            restored,
            gradient_input,
        )

        quantum_gradient = gradients[
            "quantum_weights"
        ]

        print(
            "Serialized bytes          :",
            serialized_bytes,
        )
        print(
            "Missing keys              :",
            load_result.missing_keys,
        )
        print(
            "Unexpected keys           :",
            load_result.unexpected_keys,
        )
        print(
            "Exact state tensors       :",
            f"{state_comparison['exact']}/"
            f"{state_comparison['total']}",
        )
        print(
            "Max state difference      :",
            state_comparison[
                "max_abs_difference"
            ],
        )
        print(
            "Output exact              :",
            output_exact,
        )
        print(
            "Max output difference     :",
            output_max_difference,
        )
        print(
            "Quantum requires_grad     :",
            quantum_gradient[
                "requires_grad"
            ],
        )
        print(
            "Quantum gradient present  :",
            quantum_gradient[
                "gradient_present"
            ],
        )
        print(
            "Quantum gradient finite   :",
            quantum_gradient[
                "gradient_finite"
            ],
        )

        assert not load_result.missing_keys
        assert not load_result.unexpected_keys

        assert (
            state_comparison[
                "exact"
            ]
            == state_comparison[
                "total"
            ]
        )

        assert (
            state_comparison[
                "max_abs_difference"
            ]
            == 0.0
        )

        assert output_exact
        assert output_max_difference == 0.0

        if baseline_id == "B6":
            assert not quantum_gradient[
                "requires_grad"
            ]
            assert not quantum_gradient[
                "gradient_present"
            ]
        else:
            assert quantum_gradient[
                "requires_grad"
            ]
            assert quantum_gradient[
                "gradient_present"
            ]
            assert quantum_gradient[
                "gradient_finite"
            ]
            assert (
                quantum_gradient[
                    "gradient_norm"
                ]
                > 0.0
            )

        report[baseline_id] = {
            "serialized_bytes": (
                serialized_bytes
            ),
            "state_comparison": (
                state_comparison
            ),
            "output_exact": output_exact,
            "max_output_difference": (
                output_max_difference
            ),
            "gradients": gradients,
        }

    output_path = (
        root
        / "outputs/inventory/"
        "pqfg_state_dict_roundtrip.json"
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path.write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )

    print()
    print("Report:", output_path)
    print(
        "PQFG STATE_DICT ROUNDTRIP: PASSED"
    )


if __name__ == "__main__":
    main()
