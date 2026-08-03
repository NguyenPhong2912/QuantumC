from __future__ import annotations

import gc
from pathlib import Path

import torch

from src.qvisionframe.detection_trainer import (
    QVisionDetectionTrainer,
)
from src.qvisionframe.gated_detection_model import (
    GatedDetectionModel,
)


EXPECTED = {
    "B0": {
        "gate_type": "NoneType",
        "total": 2593740,
        "added": 0,
        "state_keys": 0,
    },
    "B1": {
        "gate_type": "LinearClassicalGate",
        "total": 2725069,
        "added": 131329,
        "state_keys": 3,
    },
    "B2": {
        "gate_type": "ChannelAttentionClassicalGate",
        "total": 2602205,
        "added": 8465,
        "state_keys": 5,
    },
    "B3": {
        "gate_type": "ClassicalMLPGate",
        "total": 2628189,
        "added": 34449,
        "state_keys": 11,
    },
    "B4": {
        "gate_type": "ParameterMatchedClassicalGate",
        "total": 2628129,
        "added": 34389,
        "state_keys": 11,
    },
    "B5": {
        "gate_type": "TrigonometricClassicalGate",
        "total": 2628153,
        "added": 34413,
        "state_keys": 11,
    },
}


def count_parameters(
    model: torch.nn.Module,
) -> int:
    return sum(
        parameter.numel()
        for parameter in model.parameters()
    )


def optimizer_visibility(
    model: GatedDetectionModel,
) -> dict[str, bool]:
    # This reproduces the parameter discovery used before optimizer
    # construction: every trainable named parameter must be visible.
    return {
        name: parameter.requires_grad
        for name, parameter in model.named_parameters()
        if name.startswith("gate.")
    }


def main() -> None:
    print("===== QVISION TRAINER B0-B5 MATRIX CHECK =====")

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")

    root = Path(__file__).resolve().parents[1]
    device = torch.device("cuda:0")

    common_overrides = {
        "model": "yolo11n.yaml",
        "data": str(
            root / "configs/datasets/voc.yaml"
        ),
        "epochs": 1,
        "imgsz": 320,
        "batch": 2,
        "device": 0,
        "workers": 0,
        "fraction": 0.01,
        "pretrained": False,
        "amp": False,
        "plots": False,
        "save": False,
        "val": False,
        "project": str(
            root
            / "outputs/FITLAB-01/trainer_matrix"
        ),
        "exist_ok": True,
        "verbose": False,
    }

    for baseline_id, expected in EXPECTED.items():
        overrides = dict(common_overrides)
        overrides["name"] = (
            f"{baseline_id.lower()}_init_check"
        )

        trainer = QVisionDetectionTrainer(
            baseline_id=baseline_id,
            target_layer=10,
            gate_channels=256,
            latent_dim=4,
            hidden_dim=64,
            transform_hidden_dim=8,
            n_layers=2,
            reduction=16,
            alpha_init=1.0e-3,
            overrides=overrides,
        )

        model = trainer.get_model(
            cfg=trainer.args.model,
            weights=None,
            verbose=False,
        ).to(device).eval()

        assert isinstance(
            model,
            GatedDetectionModel,
        )

        model.reset_gate_observations()

        with torch.no_grad():
            output = model(
                torch.rand(
                    1,
                    3,
                    320,
                    320,
                    device=device,
                )
            )

        total_parameters = count_parameters(model)
        added_parameters = (
            total_parameters - 2593740
        )

        gate_type = type(model.gate).__name__

        gate_state_keys = [
            key
            for key in model.state_dict()
            if key.startswith("gate.")
        ]

        gate_parameter_visibility = (
            optimizer_visibility(model)
        )

        print(
            f"{baseline_id} | "
            f"gate={gate_type:<38} | "
            f"nc={model.model[-1].nc} | "
            f"total={total_parameters} | "
            f"added={added_parameters} | "
            f"hook={model.hook_calls} | "
            f"state_keys={len(gate_state_keys)} | "
            f"output={type(output).__name__}"
        )

        assert trainer.data["nc"] == 20
        assert model.model[-1].nc == 20
        assert model.baseline_id == baseline_id
        assert gate_type == expected["gate_type"]
        assert total_parameters == expected["total"]
        assert added_parameters == expected["added"]
        assert (
            len(gate_state_keys)
            == expected["state_keys"]
        )

        if baseline_id == "B0":
            assert model.gate is None
            assert model.hook_calls == 0
            assert model.last_feature_shape is None
            assert not gate_parameter_visibility
        else:
            assert model.gate is not None
            assert model.hook_calls == 1
            assert model.last_feature_shape == (
                1,
                256,
                10,
                10,
            )
            assert gate_parameter_visibility
            assert all(
                gate_parameter_visibility.values()
            )

        model.close_gate_hook()

        del output
        del model
        del trainer

        gc.collect()
        torch.cuda.empty_cache()

    print("QVISION TRAINER B0-B5 MATRIX: PASSED")


if __name__ == "__main__":
    main()
