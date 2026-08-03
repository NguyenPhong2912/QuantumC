from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import torch
import yaml


def sha256(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for block in iter(
            lambda: handle.read(1024 * 1024),
            b"",
        ):
            digest.update(block)

    return digest.hexdigest()


def read_single_result(path: Path) -> dict[str, float]:
    with path.open(
        "r",
        newline="",
        encoding="utf-8",
    ) as handle:
        rows = list(csv.DictReader(handle))

    if len(rows) != 1:
        raise RuntimeError(
            f"Expected one result row in {path}, "
            f"received {len(rows)}"
        )

    return {
        key.strip(): float(value)
        for key, value in rows[0].items()
    }


def load_checkpoint_model_state(
    path: Path,
) -> dict[str, torch.Tensor]:
    checkpoint: Any = torch.load(
        path,
        map_location="cpu",
        weights_only=False,
    )

    if not isinstance(checkpoint, dict):
        raise TypeError(
            f"Unexpected checkpoint type: {type(checkpoint)}"
        )

    model = checkpoint.get("model")

    if model is None:
        model = checkpoint.get("ema")

    if model is None:
        raise KeyError(
            f"No model or ema object found in {path}"
        )

    model = model.float()

    return {
        key: tensor.detach().cpu()
        for key, tensor in model.state_dict().items()
    }


def compare_states(
    first: dict[str, torch.Tensor],
    second: dict[str, torch.Tensor],
) -> tuple[
    int,
    int,
    float,
    float,
]:
    if first.keys() != second.keys():
        missing_first = sorted(
            second.keys() - first.keys()
        )
        missing_second = sorted(
            first.keys() - second.keys()
        )

        raise RuntimeError(
            "Checkpoint keys differ. "
            f"Missing from first={missing_first[:5]}, "
            f"missing from second={missing_second[:5]}"
        )

    exact_tensors = 0
    total_tensors = len(first)
    max_abs_difference = 0.0
    total_squared_difference = 0.0
    total_elements = 0

    for key in first:
        first_tensor = first[key]
        second_tensor = second[key]

        if first_tensor.shape != second_tensor.shape:
            raise RuntimeError(
                f"Shape mismatch for {key}: "
                f"{tuple(first_tensor.shape)} != "
                f"{tuple(second_tensor.shape)}"
            )

        if torch.equal(first_tensor, second_tensor):
            exact_tensors += 1

        difference = (
            first_tensor.float()
            - second_tensor.float()
        )

        if difference.numel() > 0:
            max_abs_difference = max(
                max_abs_difference,
                float(
                    difference.abs().max()
                ),
            )

            total_squared_difference += float(
                difference.square().sum()
            )
            total_elements += difference.numel()

    rms_difference = (
        (
            total_squared_difference
            / total_elements
        )
        ** 0.5
        if total_elements
        else 0.0
    )

    return (
        exact_tensors,
        total_tensors,
        max_abs_difference,
        rms_difference,
    )


def main() -> None:
    print("===== B0 CROSS-SERVER REPRODUCIBILITY =====")

    project_root = Path(__file__).resolve().parents[1]

    run_01 = (
        project_root
        / "outputs/FITLAB-01/smoke_training/"
        "voc_b0_yolo11n_seed42"
    )
    run_02 = (
        project_root
        / "outputs/FITLAB-02/smoke_training/"
        "voc_b0_yolo11n_seed42"
    )

    for run_dir in (run_01, run_02):
        for required in (
            run_dir / "results.csv",
            run_dir / "args.yaml",
            run_dir / "weights/best.pt",
            run_dir / "weights/last.pt",
        ):
            if not required.is_file():
                raise FileNotFoundError(required)

    results_01 = read_single_result(
        run_01 / "results.csv"
    )
    results_02 = read_single_result(
        run_02 / "results.csv"
    )

    metric_differences = {
        key: abs(
            results_01[key]
            - results_02[key]
        )
        for key in results_01
        if key != "time"
    }

    max_metric_difference = max(
        metric_differences.values(),
        default=0.0,
    )

    args_01 = yaml.safe_load(
        (run_01 / "args.yaml").read_text(
            encoding="utf-8"
        )
    )
    args_02 = yaml.safe_load(
        (run_02 / "args.yaml").read_text(
            encoding="utf-8"
        )
    )

    ignored_args = {
        "project",
        "name",
        "save_dir",
    }

    comparable_args_01 = {
        key: value
        for key, value in args_01.items()
        if key not in ignored_args
    }
    comparable_args_02 = {
        key: value
        for key, value in args_02.items()
        if key not in ignored_args
    }

    args_equal = (
        comparable_args_01
        == comparable_args_02
    )

    best_01 = run_01 / "weights/best.pt"
    best_02 = run_02 / "weights/best.pt"

    file_hash_01 = sha256(best_01)
    file_hash_02 = sha256(best_02)

    state_01 = load_checkpoint_model_state(
        best_01
    )
    state_02 = load_checkpoint_model_state(
        best_02
    )

    (
        exact_tensors,
        total_tensors,
        max_weight_difference,
        rms_weight_difference,
    ) = compare_states(state_01, state_02)

    summary = {
        "fitlab_01": str(run_01),
        "fitlab_02": str(run_02),
        "args_equal_excluding_paths": args_equal,
        "max_metric_difference_excluding_time": (
            max_metric_difference
        ),
        "best_pt_sha256_fitlab_01": file_hash_01,
        "best_pt_sha256_fitlab_02": file_hash_02,
        "checkpoint_files_identical": (
            file_hash_01 == file_hash_02
        ),
        "exact_weight_tensors": exact_tensors,
        "total_weight_tensors": total_tensors,
        "max_weight_abs_difference": (
            max_weight_difference
        ),
        "rms_weight_difference": (
            rms_weight_difference
        ),
    }

    output_path = (
        project_root
        / "outputs/inventory/"
        "b0_cross_server_reproducibility.json"
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path.write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )

    print("Args equal             :", args_equal)
    print(
        "Max metric difference :",
        max_metric_difference,
    )
    print(
        "Checkpoint file equal :",
        file_hash_01 == file_hash_02,
    )
    print(
        "Exact weight tensors  :",
        f"{exact_tensors}/{total_tensors}",
    )
    print(
        "Max weight difference :",
        max_weight_difference,
    )
    print(
        "RMS weight difference :",
        rms_weight_difference,
    )
    print("Summary                :", output_path)

    assert args_equal
    assert max_metric_difference <= 1.0e-6
    assert max_weight_difference <= 1.0e-6

    print("B0 CROSS-SERVER REPRODUCIBILITY: PASSED")


if __name__ == "__main__":
    main()
