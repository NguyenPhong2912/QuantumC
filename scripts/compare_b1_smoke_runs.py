from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import torch
import yaml

# Import bắt buộc để torch.load có thể giải tuần tự hóa
# custom checkpoint class.
from src.qvisionframe.gated_detection_model import (
    GatedDetectionModel,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for block in iter(
            lambda: handle.read(1024 * 1024),
            b"",
        ):
            digest.update(block)

    return digest.hexdigest()


def read_result_row(path: Path) -> dict[str, float]:
    with path.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as handle:
        rows = list(csv.DictReader(handle))

    if len(rows) != 1:
        raise RuntimeError(
            f"Expected exactly one result row in {path}, "
            f"received {len(rows)}"
        )

    return {
        key.strip(): float(value)
        for key, value in rows[0].items()
    }


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(
        path.read_text(encoding="utf-8")
    )

    if not isinstance(value, dict):
        raise TypeError(
            f"Expected JSON object in {path}"
        )

    return value


def load_model_state(
    path: Path,
) -> dict[str, torch.Tensor]:
    checkpoint: Any = torch.load(
        path,
        map_location="cpu",
        weights_only=False,
    )

    if not isinstance(checkpoint, dict):
        raise TypeError(
            f"Unexpected checkpoint type: "
            f"{type(checkpoint).__name__}"
        )

    model = checkpoint.get("ema")

    if model is None:
        model = checkpoint.get("model")

    if model is None:
        raise KeyError(
            f"No ema/model object in {path}"
        )

    model = model.float()

    if not isinstance(model, GatedDetectionModel):
        raise TypeError(
            "Expected GatedDetectionModel, received "
            f"{type(model).__name__}"
        )

    return {
        key: tensor.detach().cpu().clone()
        for key, tensor in model.state_dict().items()
    }


def compare_tensor_states(
    first: dict[str, torch.Tensor],
    second: dict[str, torch.Tensor],
) -> dict[str, Any]:
    first_keys = set(first)
    second_keys = set(second)

    if first_keys != second_keys:
        raise RuntimeError(
            "State keys differ. "
            f"Only first={sorted(first_keys - second_keys)[:10]}, "
            f"only second={sorted(second_keys - first_keys)[:10]}"
        )

    exact_count = 0
    max_abs_difference = 0.0
    squared_difference_sum = 0.0
    total_elements = 0
    nonexact_keys: list[str] = []

    for key in sorted(first):
        tensor_01 = first[key]
        tensor_02 = second[key]

        if tensor_01.shape != tensor_02.shape:
            raise RuntimeError(
                f"Shape mismatch for {key}: "
                f"{tuple(tensor_01.shape)} != "
                f"{tuple(tensor_02.shape)}"
            )

        exact = torch.equal(
            tensor_01,
            tensor_02,
        )

        if exact:
            exact_count += 1
        else:
            nonexact_keys.append(key)

        difference = (
            tensor_01.float()
            - tensor_02.float()
        )

        if difference.numel():
            max_abs_difference = max(
                max_abs_difference,
                float(difference.abs().max()),
            )

            squared_difference_sum += float(
                difference.square().sum()
            )

            total_elements += difference.numel()

    rms_difference = (
        (
            squared_difference_sum
            / total_elements
        )
        ** 0.5
        if total_elements
        else 0.0
    )

    return {
        "exact_tensors": exact_count,
        "total_tensors": len(first),
        "nonexact_keys": nonexact_keys,
        "max_abs_difference": max_abs_difference,
        "rms_difference": rms_difference,
    }


def main() -> None:
    print("===== B1 CROSS-SERVER REPRODUCIBILITY =====")

    root = Path(__file__).resolve().parents[1]

    run_01 = (
        root
        / "outputs/FITLAB-01/smoke_training/"
        "voc_b1_linear_seed42"
    )

    run_02 = (
        root
        / "outputs/FITLAB-02/smoke_training/"
        "voc_b1_linear_seed42"
    )

    required_relative_paths = (
        Path("results.csv"),
        Path("args.yaml"),
        Path("b1_smoke_summary.json"),
        Path("weights/best.pt"),
        Path("weights/last.pt"),
    )

    for run_dir in (run_01, run_02):
        for relative_path in required_relative_paths:
            path = run_dir / relative_path

            if not path.is_file():
                raise FileNotFoundError(path)

    # --------------------------------------------------------
    # 1. Training arguments
    # --------------------------------------------------------
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

    # --------------------------------------------------------
    # 2. Results
    # --------------------------------------------------------
    results_01 = read_result_row(
        run_01 / "results.csv"
    )

    results_02 = read_result_row(
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

    # --------------------------------------------------------
    # 3. Smoke summaries
    # --------------------------------------------------------
    summary_01 = read_json(
        run_01 / "b1_smoke_summary.json"
    )

    summary_02 = read_json(
        run_02 / "b1_smoke_summary.json"
    )

    optimizer_equal = (
        summary_01["gate_in_optimizer"]
        == summary_02["gate_in_optimizer"]
    )

    gate_update_equal = (
        summary_01["gate_state_changes"]
        == summary_02["gate_state_changes"]
    )

    hook_count_equal = (
        summary_01["final_hook_calls"]
        == summary_02["final_hook_calls"]
    )

    # --------------------------------------------------------
    # 4. Checkpoint state
    # --------------------------------------------------------
    best_01 = run_01 / "weights/best.pt"
    best_02 = run_02 / "weights/best.pt"

    checkpoint_hash_01 = sha256(best_01)
    checkpoint_hash_02 = sha256(best_02)

    state_01 = load_model_state(best_01)
    state_02 = load_model_state(best_02)

    state_comparison = compare_tensor_states(
        state_01,
        state_02,
    )

    gate_state_01 = {
        key: value
        for key, value in state_01.items()
        if key.startswith("gate.")
    }

    gate_state_02 = {
        key: value
        for key, value in state_02.items()
        if key.startswith("gate.")
    }

    gate_comparison = compare_tensor_states(
        gate_state_01,
        gate_state_02,
    )

    expected_gate_keys = {
        "gate.alpha",
        "gate.linear.weight",
        "gate.linear.bias",
    }

    summary = {
        "fitlab_01": str(run_01),
        "fitlab_02": str(run_02),
        "args_equal_excluding_paths": args_equal,
        "max_metric_difference_excluding_time": (
            max_metric_difference
        ),
        "metric_differences": metric_differences,
        "optimizer_registration_equal": (
            optimizer_equal
        ),
        "gate_update_summary_equal": (
            gate_update_equal
        ),
        "hook_count_equal": hook_count_equal,
        "checkpoint_sha256_fitlab_01": (
            checkpoint_hash_01
        ),
        "checkpoint_sha256_fitlab_02": (
            checkpoint_hash_02
        ),
        "checkpoint_files_identical": (
            checkpoint_hash_01
            == checkpoint_hash_02
        ),
        "full_model_state": state_comparison,
        "gate_state": gate_comparison,
        "gate_keys": sorted(gate_state_01),
    }

    output_path = (
        root
        / "outputs/inventory/"
        "b1_cross_server_reproducibility.json"
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path.write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )

    print("Args equal                 :", args_equal)
    print(
        "Max metric difference      :",
        max_metric_difference,
    )
    print(
        "Optimizer audit equal       :",
        optimizer_equal,
    )
    print(
        "Gate update audit equal     :",
        gate_update_equal,
    )
    print(
        "Hook count equal            :",
        hook_count_equal,
    )
    print(
        "Checkpoint file hash equal  :",
        checkpoint_hash_01 == checkpoint_hash_02,
    )
    print(
        "Exact model tensors         :",
        f"{state_comparison['exact_tensors']}/"
        f"{state_comparison['total_tensors']}",
    )
    print(
        "Max model weight difference :",
        state_comparison["max_abs_difference"],
    )
    print(
        "RMS model weight difference :",
        state_comparison["rms_difference"],
    )
    print(
        "Exact gate tensors          :",
        f"{gate_comparison['exact_tensors']}/"
        f"{gate_comparison['total_tensors']}",
    )
    print(
        "Max gate weight difference  :",
        gate_comparison["max_abs_difference"],
    )
    print("Summary                     :", output_path)

    assert args_equal
    assert max_metric_difference <= 1.0e-6
    assert optimizer_equal
    assert gate_update_equal
    assert hook_count_equal

    assert set(gate_state_01) == expected_gate_keys
    assert set(gate_state_02) == expected_gate_keys

    assert (
        state_comparison["max_abs_difference"]
        <= 1.0e-6
    )

    assert (
        gate_comparison["max_abs_difference"]
        <= 1.0e-6
    )

    print("B1 CROSS-SERVER REPRODUCIBILITY: PASSED")


if __name__ == "__main__":
    main()
