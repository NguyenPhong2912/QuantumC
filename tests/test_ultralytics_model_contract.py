from __future__ import annotations

import inspect
from dataclasses import dataclass

import torch

from src.qvisionframe.baseline_factory import (
    BaselineBuildConfig,
    build_baseline,
)


@dataclass(frozen=True)
class ContractResult:
    baseline_id: str
    has_model: bool
    has_stride: bool
    has_names: bool
    has_yaml: bool
    has_args: bool
    has_criterion: bool
    has_loss: bool
    has_load: bool
    loss_callable: bool
    load_callable: bool


REQUIRED_ATTRIBUTES = (
    "model",
    "stride",
    "names",
    "yaml",
    "args",
    "criterion",
    "loss",
    "load",
)


def detector_of(model):
    return model if not hasattr(model, "yolo") else model.yolo


def close_model(model) -> None:
    close = getattr(model, "close", None)

    if callable(close):
        close()


def audit(
    baseline_id: str,
) -> ContractResult:
    model = build_baseline(
        BaselineBuildConfig(
            baseline_id=baseline_id,
            architecture="yolo11n.yaml",
            num_classes=20,
            device="cuda:0",
        )
    )

    result = ContractResult(
        baseline_id=baseline_id,
        has_model=hasattr(model, "model"),
        has_stride=hasattr(model, "stride"),
        has_names=hasattr(model, "names"),
        has_yaml=hasattr(model, "yaml"),
        has_args=hasattr(model, "args"),
        has_criterion=hasattr(model, "criterion"),
        has_loss=hasattr(model, "loss"),
        has_load=hasattr(model, "load"),
        loss_callable=callable(
            getattr(model, "loss", None)
        ),
        load_callable=callable(
            getattr(model, "load", None)
        ),
    )

    detector = detector_of(model)

    print(
        f"{baseline_id} | "
        f"type={type(model).__name__:<34} | "
        f"detector={type(detector).__name__}"
    )

    for attribute in REQUIRED_ATTRIBUTES:
        value = getattr(model, attribute, None)

        print(
            f"     {attribute:<10}: "
            f"{hasattr(model, attribute)!s:<5} "
            f"type={type(value).__name__}"
        )

    print(
        "     detector nc:",
        detector.model[-1].nc,
    )
    print(
        "     forward signature:",
        inspect.signature(model.forward),
    )
    print("-" * 88)

    close_model(model)
    del model
    torch.cuda.empty_cache()

    return result


def main() -> None:
    print("===== ULTRALYTICS MODEL CONTRACT AUDIT =====")

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")

    results = [
        audit(baseline_id)
        for baseline_id in (
            "B0",
            "B1",
            "B2",
            "B3",
            "B4",
            "B5",
            "B6",
            "B7",
            "B8",
        )
    ]

    b0 = results[0]

    # Constructor-level DetectionModel contract.
    #
    # `args` is normally assigned by the trainer, while `criterion`
    # may be initialized lazily by DetectionModel.loss(). Therefore,
    # neither attribute is required immediately after construction.
    assert b0.has_model
    assert b0.has_stride
    assert b0.has_names
    assert b0.has_yaml
    assert b0.has_loss
    assert b0.has_load
    assert b0.loss_callable
    assert b0.load_callable

    incompatible = [
        result.baseline_id
        for result in results[1:]
        if not all(
            (
                result.has_model,
                result.has_stride,
                result.has_names,
                result.has_yaml,
                result.has_loss,
                result.has_load,
                result.loss_callable,
                result.load_callable,
            )
        )
    ]

    print("B0 constructor contract: PASSED")
    print(
        "B0 runtime state       :",
        {
            "args_initialized": b0.has_args,
            "criterion_initialized": b0.has_criterion,
        },
    )
    print("Wrappers incompatible :", incompatible)

    # Existing external wrappers are expected to be incomplete.
    assert incompatible == [
        "B1",
        "B2",
        "B3",
        "B4",
        "B5",
        "B6",
        "B7",
        "B8",
    ]

    print("CONTRACT AUDIT        : PASSED")
    print(
        "NEXT ACTION           : build a DetectionModel "
        "subclass with internal gate injection"
    )


if __name__ == "__main__":
    main()
