from __future__ import annotations

import gc

import torch
from ultralytics.cfg import get_cfg
from ultralytics.utils import DEFAULT_CFG

from src.qvisionframe.quantum_detection_model import (
    QuantumGatedDetectionModel,
)


EXPECTED = {
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


def count_parameters(
    model: torch.nn.Module,
) -> int:
    return sum(
        parameter.numel()
        for parameter in model.parameters()
    )


def count_trainable(
    model: torch.nn.Module,
) -> int:
    return sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )


def make_batch(
    device: torch.device,
) -> dict[str, torch.Tensor]:
    return {
        "img": torch.rand(
            2,
            3,
            320,
            320,
            device=device,
        ),
        "cls": torch.tensor(
            [[0.0], [5.0], [2.0], [7.0]],
            dtype=torch.float32,
            device=device,
        ),
        "bboxes": torch.tensor(
            [
                [0.30, 0.35, 0.20, 0.25],
                [0.70, 0.65, 0.15, 0.20],
                [0.40, 0.45, 0.25, 0.30],
                [0.75, 0.30, 0.18, 0.22],
            ],
            dtype=torch.float32,
            device=device,
        ),
        "batch_idx": torch.tensor(
            [0, 0, 1, 1],
            dtype=torch.long,
            device=device,
        ),
    }


def gradient_information(
    parameter: torch.nn.Parameter,
) -> dict[str, object]:
    gradient = parameter.grad

    return {
        "requires_grad": (
            parameter.requires_grad
        ),
        "present": gradient is not None,
        "finite": (
            bool(torch.isfinite(gradient).all())
            if gradient is not None
            else None
        ),
        "norm": (
            float(
                gradient.detach()
                .float()
                .norm()
                .cpu()
            )
            if gradient is not None
            else None
        ),
        "dtype": (
            str(gradient.dtype)
            if gradient is not None
            else None
        ),
        "device": (
            str(gradient.device)
            if gradient is not None
            else None
        ),
    }


def main() -> None:
    print(
        "===== QUANTUM DETECTION MODEL B6-B8 ====="
    )

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is unavailable"
        )

    device = torch.device("cuda:0")

    for baseline_id, expected in EXPECTED.items():
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)

        model = QuantumGatedDetectionModel(
            cfg="yolo11n.yaml",
            nc=20,
            baseline_id=baseline_id,
            target_layer=10,
            channels=256,
            n_qubits=4,
            n_layers=2,
            hidden_dim=64,
            alpha_init=1.0e-3,
            gamma=0.5,
            verbose=False,
        ).to(device).train()

        model.args = get_cfg(
            cfg=DEFAULT_CFG
        )
        model.criterion = None

        model.zero_grad(
            set_to_none=True
        )
        model.reset_gate_observations()

        batch = make_batch(device)

        loss_components, loss_items = model(
            batch
        )

        total_loss = loss_components.sum()
        total_loss.backward()

        total_parameters = count_parameters(
            model
        )

        trainable_parameters = count_trainable(
            model
        )

        state_gate_keys = [
            key
            for key in model.state_dict()
            if key.startswith("gate.")
        ]

        quantum_gradient = gradient_information(
            model.gate.quantum_weights
        )

        alpha_gradient = gradient_information(
            model.gate.alpha
        )

        print()
        print(baseline_id)
        print(
            "  total parameters       :",
            total_parameters,
        )
        print(
            "  trainable parameters   :",
            trainable_parameters,
        )
        print(
            "  gate state keys        :",
            len(state_gate_keys),
        )
        print(
            "  hook calls             :",
            model.hook_calls,
        )
        print(
            "  feature shape          :",
            model.last_feature_shape,
        )
        print(
            "  quantum dtype/device   :",
            model.gate.quantum_weights.dtype,
            model.gate.quantum_weights.device,
        )
        print(
            "  quantum gradient       :",
            quantum_gradient,
        )
        print(
            "  alpha gradient         :",
            alpha_gradient,
        )
        print(
            "  loss components        :",
            loss_components.detach()
            .float()
            .cpu()
            .tolist(),
        )

        assert model.baseline_id == baseline_id
        assert model.model[-1].nc == 20

        assert (
            model.qvf_gate_config[
                "entanglement"
            ]
            == expected["entanglement"]
        )

        assert (
            model.qvf_gate_config[
                "trainable_quantum"
            ]
            == expected["trainable_quantum"]
        )

        assert total_parameters == 2628129
        assert len(state_gate_keys) == 8

        assert model.hook_calls == 1
        assert model.last_feature_shape == (
            2,
            256,
            10,
            10,
        )

        assert (
            model.gate.quantum_weights.dtype
            == torch.float64
        )

        assert (
            model.gate.quantum_weights.device.type
            == "cuda"
        )

        assert torch.isfinite(
            loss_components
        ).all()

        assert torch.isfinite(
            loss_items
        ).all()

        assert float(total_loss.detach()) > 0.0

        assert alpha_gradient["present"]
        assert alpha_gradient["finite"]
        assert alpha_gradient["norm"] > 0.0

        if baseline_id == "B6":
            assert (
                quantum_gradient[
                    "requires_grad"
                ]
                is False
            )
            assert (
                quantum_gradient["present"]
                is False
            )
        else:
            assert (
                quantum_gradient[
                    "requires_grad"
                ]
                is True
            )
            assert quantum_gradient["present"]
            assert quantum_gradient["finite"]
            assert (
                quantum_gradient["norm"]
                > 0.0
            )
            assert (
                quantum_gradient["dtype"]
                == "torch.float64"
            )
            assert (
                quantum_gradient["device"]
                == "cuda:0"
            )

        metadata = (
            model.quantum_checkpoint_metadata()
        )

        assert metadata["baseline_id"] == (
            baseline_id
        )

        model.close_gate_hook()

        del batch
        del loss_components
        del loss_items
        del total_loss
        del model

        gc.collect()
        torch.cuda.empty_cache()

    print()
    print(
        "QUANTUM DETECTION MODEL B6-B8: PASSED"
    )


if __name__ == "__main__":
    main()
