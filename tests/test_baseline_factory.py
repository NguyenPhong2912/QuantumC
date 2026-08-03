from __future__ import annotations

import torch

from src.qvisionframe.baseline_factory import (
    BaselineBuildConfig,
    build_baseline,
)


def parameter_count(module: torch.nn.Module) -> int:
    return sum(
        parameter.numel()
        for parameter in module.parameters()
    )


def close_model(model) -> None:
    close = getattr(model, "close", None)
    if callable(close):
        close()


def main() -> None:
    print("===== UNIFIED BASELINE FACTORY CHECK =====")

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")

    device = torch.device("cuda:0")

    expected_added = {
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

    detector_parameters = None

    for baseline_id in expected_added:
        torch.manual_seed(42)

        model = build_baseline(
            BaselineBuildConfig(
                baseline_id=baseline_id,
                architecture="yolo11n.yaml",
                device="cuda:0",
            )
        ).eval()

        current_total = parameter_count(model)

        if baseline_id == "B0":
            detector_parameters = current_total
            added_parameters = 0
        else:
            if detector_parameters is None:
                raise RuntimeError(
                    "B0 must be constructed first"
                )

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
                    dtype=torch.float32,
                    device=device,
                )
            )

        output_type = type(output).__name__

        print(
            f"{baseline_id} | "
            f"total={current_total} | "
            f"added={added_parameters} | "
            f"output={output_type}"
        )

        assert added_parameters == expected_added[baseline_id]

        if baseline_id != "B0":
            assert model.hook_calls == 1
            assert model.last_feature_shape == (
                1,
                256,
                10,
                10,
            )

        if baseline_id == "B6":
            assert (
                model.pqfg.quantum_weights.requires_grad
                is False
            )

        if baseline_id in {"B7", "B8"}:
            assert (
                model.pqfg.quantum_weights.requires_grad
                is True
            )

        close_model(model)

        del model
        torch.cuda.empty_cache()

    assert detector_parameters == 2624080

    print("UNIFIED BASELINE FACTORY: PASSED")


if __name__ == "__main__":
    main()
