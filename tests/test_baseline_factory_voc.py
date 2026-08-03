from __future__ import annotations

import torch

from src.qvisionframe.baseline_factory import (
    BaselineBuildConfig,
    build_baseline,
)


EXPECTED_ADDED = {
    "B0": 0,
    "B1": 131329,
    "B2": 8465,
    "B3": 34449,
    "B4": 34389,
    "B5": 34413,
    "B6": 34389,
    "B7": 34389,
    "B8": 34389,
}


def parameter_count(module: torch.nn.Module) -> int:
    return sum(
        parameter.numel()
        for parameter in module.parameters()
    )


def get_detector(model):
    return model if not hasattr(model, "yolo") else model.yolo


def close_model(model) -> None:
    close = getattr(model, "close", None)
    if callable(close):
        close()


def main() -> None:
    print("===== VOC BASELINE FACTORY CHECK =====")

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")

    detector_parameters = None

    for baseline_id, expected_added in EXPECTED_ADDED.items():
        torch.manual_seed(42)

        model = build_baseline(
            BaselineBuildConfig(
                baseline_id=baseline_id,
                architecture="yolo11n.yaml",
                num_classes=20,
                device="cuda:0",
            )
        ).eval()

        detector = get_detector(model)
        detect_head = detector.model[-1]

        current_total = parameter_count(model)

        if baseline_id == "B0":
            detector_parameters = current_total
            added_parameters = 0
        else:
            if detector_parameters is None:
                raise RuntimeError("B0 must be checked first")

            added_parameters = (
                current_total - detector_parameters
            )

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

        print(
            f"{baseline_id} | "
            f"nc={detect_head.nc} | "
            f"total={current_total} | "
            f"added={added_parameters} | "
            f"output={type(output).__name__}"
        )

        assert detect_head.nc == 20
        assert added_parameters == expected_added

        if baseline_id != "B0":
            assert model.hook_calls == 1
            assert model.last_feature_shape == (
                1,
                256,
                10,
                10,
            )

        close_model(model)
        del model
        torch.cuda.empty_cache()

    assert detector_parameters == 2593740

    print("VOC DETECTOR PARAMETERS:", detector_parameters)
    print("VOC BASELINE FACTORY   : PASSED")


if __name__ == "__main__":
    main()
