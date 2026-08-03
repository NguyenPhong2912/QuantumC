from __future__ import annotations

from pathlib import Path

import torch

from src.qvisionframe.detection_trainer import (
    QVisionDetectionTrainer,
)
from src.qvisionframe.gated_detection_model import (
    GatedDetectionModel,
)


def count_parameters(module: torch.nn.Module) -> int:
    return sum(
        parameter.numel()
        for parameter in module.parameters()
    )


def main() -> None:
    print("===== QVISION DETECTION TRAINER INIT CHECK =====")

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")

    root = Path(__file__).resolve().parents[1]

    overrides = {
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
            root / "outputs/FITLAB-01/trainer_init"
        ),
        "name": "b1_init_check",
        "exist_ok": True,
        "verbose": False,
    }

    trainer = QVisionDetectionTrainer(
        baseline_id="B1",
        target_layer=10,
        gate_channels=256,
        alpha_init=1.0e-3,
        overrides=overrides,
    )

    model = trainer.get_model(
        cfg=trainer.args.model,
        weights=None,
        verbose=False,
    ).cuda()

    assert isinstance(model, GatedDetectionModel)
    assert model.baseline_id == "B1"
    assert model.model[-1].nc == 20
    assert model.gate is not None

    total_parameters = count_parameters(model)

    gate_parameter_names = [
        name
        for name, _ in model.named_parameters()
        if name.startswith("gate.")
    ]

    trainable_gate_parameters = [
        name
        for name, parameter in model.named_parameters()
        if name.startswith("gate.")
        and parameter.requires_grad
    ]

    with torch.no_grad():
        output = model(
            torch.rand(
                1,
                3,
                320,
                320,
                device="cuda:0",
            )
        )

    print("Trainer type          :", type(trainer).__name__)
    print("Model type            :", type(model).__name__)
    print("Baseline              :", model.baseline_id)
    print("Dataset classes       :", trainer.data["nc"])
    print("Detection head classes:", model.model[-1].nc)
    print("Total parameters      :", total_parameters)
    print("Feature shape         :", model.last_feature_shape)
    print("Hook calls            :", model.hook_calls)
    print("Output type           :", type(output).__name__)
    print("Gate parameters       :", gate_parameter_names)
    print(
        "Trainable gate params :",
        trainable_gate_parameters,
    )

    assert trainer.data["nc"] == 20
    assert total_parameters == 2725069
    assert model.last_feature_shape == (
        1,
        256,
        10,
        10,
    )
    assert model.hook_calls == 1

    assert gate_parameter_names == [
        "gate.alpha",
        "gate.linear.weight",
        "gate.linear.bias",
    ]

    assert trainable_gate_parameters == (
        gate_parameter_names
    )

    model.close_gate_hook()

    print("QVISION DETECTION TRAINER INIT: PASSED")


if __name__ == "__main__":
    main()
