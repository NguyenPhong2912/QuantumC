from __future__ import annotations

import gc

import torch
from ultralytics.cfg import get_cfg
from ultralytics.utils import DEFAULT_CFG

from src.qvisionframe.gated_detection_model import (
    GatedDetectionModel,
)


EXPECTED_ADDED_PARAMETERS = {
    "B0": 0,
    "B1": 131329,
    "B2": 8465,
    "B3": 34449,
    "B4": 34389,
    "B5": 34413,
}


def parameter_count(
    module: torch.nn.Module,
) -> int:
    return sum(
        parameter.numel()
        for parameter in module.parameters()
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


def gate_gradient_norm(
    model: GatedDetectionModel,
) -> float:
    if model.gate is None:
        return 0.0

    squared_sum = 0.0

    for parameter in model.gate.parameters():
        if parameter.grad is None:
            continue

        gradient = parameter.grad.detach().float()
        squared_sum += float(
            gradient.square().sum().cpu()
        )

    return squared_sum ** 0.5


def main() -> None:
    print(
        "===== TRAINER-COMPATIBLE B0-B5 CHECK ====="
    )

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is unavailable"
        )

    device = torch.device("cuda:0")

    torch.manual_seed(42)

    reference = GatedDetectionModel(
        cfg="yolo11n.yaml",
        nc=20,
        baseline_id="B0",
        verbose=False,
    ).to(device)

    detector_parameters = parameter_count(reference)

    assert detector_parameters == 2593740

    reference.close_gate_hook()
    del reference
    torch.cuda.empty_cache()

    for baseline_id, expected_added in (
        EXPECTED_ADDED_PARAMETERS.items()
    ):
        torch.manual_seed(42)

        model = GatedDetectionModel(
            cfg="yolo11n.yaml",
            nc=20,
            baseline_id=baseline_id,
            target_layer=10,
            channels=256,
            latent_dim=4,
            hidden_dim=64,
            transform_hidden_dim=8,
            n_layers=2,
            reduction=16,
            alpha_init=1.0e-3,
            verbose=False,
        ).to(device).train()

        model.args = get_cfg(
            cfg=DEFAULT_CFG
        )
        model.criterion = None
        model.zero_grad(set_to_none=True)
        model.reset_gate_observations()

        total_parameters = parameter_count(model)
        added_parameters = (
            total_parameters
            - detector_parameters
        )

        batch = make_batch(device)

        loss_components, loss_items = model(batch)
        total_loss = loss_components.sum()
        total_loss.backward()

        state_gate_keys = [
            key
            for key in model.state_dict()
            if key.startswith("gate.")
        ]

        named_gate_parameters = [
            name
            for name, _ in model.named_parameters()
            if name.startswith("gate.")
        ]

        gradient_norm = gate_gradient_norm(model)

        print(
            f"{baseline_id} | "
            f"gate={type(model.gate).__name__:<38} | "
            f"total={total_parameters} | "
            f"added={added_parameters} | "
            f"hook={model.hook_calls} | "
            f"grad={gradient_norm:.12g} | "
            f"state_keys={len(state_gate_keys)}"
        )

        assert model.model[-1].nc == 20
        assert added_parameters == expected_added

        assert torch.isfinite(
            loss_components
        ).all()

        assert torch.isfinite(
            loss_items
        ).all()

        assert float(total_loss.detach()) > 0.0

        if baseline_id == "B0":
            assert model.gate is None
            assert model.hook_calls == 0
            assert model.last_feature_shape is None
            assert not state_gate_keys
            assert not named_gate_parameters
            assert gradient_norm == 0.0
        else:
            assert model.gate is not None
            assert model.hook_calls == 1
            assert model.last_feature_shape == (
                2,
                256,
                10,
                10,
            )
            assert state_gate_keys
            assert named_gate_parameters
            assert gradient_norm > 0.0

            assert model.qvf_gate_config[
                "baseline_id"
            ] == baseline_id

        model.close_gate_hook()

        del batch
        del loss_components
        del loss_items
        del total_loss
        del model

        gc.collect()
        torch.cuda.empty_cache()

    print(
        "TRAINER-COMPATIBLE B0-B5: PASSED"
    )


if __name__ == "__main__":
    main()
