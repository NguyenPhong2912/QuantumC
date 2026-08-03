from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import torch

from src.qvisionframe.quantum_checkpoint import (
    load_quantum_checkpoint,
)


ROOT = Path(__file__).resolve().parents[1]

CHECKPOINT_01 = (
    ROOT
    / "outputs/FITLAB-01/resume_training/"
    "voc_b8_quantum_resume_seed42/"
    "weights/last.pt"
)

CHECKPOINT_02 = (
    ROOT
    / "outputs/FITLAB-02/resume_training/"
    "voc_b8_quantum_resume_seed42/"
    "weights/last.pt"
)

REPORT_PATH = (
    ROOT
    / "outputs/inventory/"
    "cross_server_b8_resume_equivalence.json"
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as stream:
        for block in iter(
            lambda: stream.read(1024 * 1024),
            b"",
        ):
            digest.update(block)

    return digest.hexdigest()


def compare_tensor_dict(
    first: dict[str, torch.Tensor],
    second: dict[str, torch.Tensor],
) -> dict[str, Any]:
    if first.keys() != second.keys():
        missing_01 = sorted(
            set(second).difference(first)
        )
        missing_02 = sorted(
            set(first).difference(second)
        )

        raise RuntimeError(
            "State keys differ. "
            f"Missing from 01={missing_01}, "
            f"missing from 02={missing_02}"
        )

    exact = 0
    nonexact: list[str] = []
    max_abs_difference = 0.0

    for key in first:
        tensor_01 = first[key]
        tensor_02 = second[key]

        is_exact = (
            tensor_01.dtype == tensor_02.dtype
            and tensor_01.shape == tensor_02.shape
            and torch.equal(
                tensor_01,
                tensor_02,
            )
        )

        if is_exact:
            exact += 1
        else:
            nonexact.append(key)

        if (
            tensor_01.shape == tensor_02.shape
            and tensor_01.numel() > 0
            and tensor_01.is_floating_point()
            and tensor_02.is_floating_point()
        ):
            difference = (
                tensor_01.to(torch.float64)
                - tensor_02.to(torch.float64)
            )

            max_abs_difference = max(
                max_abs_difference,
                float(
                    difference.abs().max()
                ),
            )

    return {
        "exact_tensors": exact,
        "total_tensors": len(first),
        "nonexact_keys": nonexact,
        "max_abs_difference": (
            max_abs_difference
        ),
    }


def compare_tree(
    first: Any,
    second: Any,
    path: str = "root",
) -> list[str]:
    """
    Return paths whose values are not exactly equal.
    """
    differences: list[str] = []

    if isinstance(first, torch.Tensor):
        if not isinstance(second, torch.Tensor):
            return [path]

        if (
            first.dtype != second.dtype
            or first.shape != second.shape
            or not torch.equal(first, second)
        ):
            differences.append(path)

        return differences

    if isinstance(first, dict):
        if not isinstance(second, dict):
            return [path]

        if first.keys() != second.keys():
            differences.append(
                f"{path}.__keys__"
            )

        common_keys = (
            first.keys() & second.keys()
        )

        for key in common_keys:
            differences.extend(
                compare_tree(
                    first[key],
                    second[key],
                    f"{path}.{key}",
                )
            )

        return differences

    if isinstance(first, (list, tuple)):
        if not isinstance(
            second,
            type(first),
        ):
            return [path]

        if len(first) != len(second):
            differences.append(
                f"{path}.__length__"
            )

        for index, (
            item_01,
            item_02,
        ) in enumerate(
            zip(first, second)
        ):
            differences.extend(
                compare_tree(
                    item_01,
                    item_02,
                    f"{path}[{index}]",
                )
            )

        return differences

    if first != second:
        differences.append(path)

    return differences


def main() -> None:
    print(
        "===== CROSS-SERVER B8 RESUME "
        "EQUIVALENCE AUDIT ====="
    )

    print("FITLAB-01 checkpoint:", CHECKPOINT_01)
    print("FITLAB-02 checkpoint:", CHECKPOINT_02)
    print("FITLAB-01 exists    :", CHECKPOINT_01.exists())
    print("FITLAB-02 exists    :", CHECKPOINT_02.exists())

    assert CHECKPOINT_01.exists()
    assert CHECKPOINT_02.exists()

    payload_01 = load_quantum_checkpoint(
        CHECKPOINT_01,
        map_location="cpu",
    )

    payload_02 = load_quantum_checkpoint(
        CHECKPOINT_02,
        map_location="cpu",
    )

    raw_comparison = compare_tensor_dict(
        payload_01["model_state_dict"],
        payload_02["model_state_dict"],
    )

    ema_comparison = compare_tensor_dict(
        payload_01["ema_state_dict"],
        payload_02["ema_state_dict"],
    )

    optimizer_differences = compare_tree(
        payload_01["optimizer_state_dict"],
        payload_02["optimizer_state_dict"],
        "optimizer",
    )

    scaler_differences = compare_tree(
        payload_01["scaler_state_dict"],
        payload_02["scaler_state_dict"],
        "scaler",
    )

    metadata_fields = (
        "qvf_format",
        "format_version",
        "epoch",
        "best_fitness",
        "ema_updates",
        "model_metadata",
        "train_metrics",
    )

    metadata_differences = []

    for key in metadata_fields:
        metadata_differences.extend(
            compare_tree(
                payload_01.get(key),
                payload_02.get(key),
                key,
            )
        )

    sha256_01 = file_sha256(
        CHECKPOINT_01
    )
    sha256_02 = file_sha256(
        CHECKPOINT_02
    )

    print()
    print("===== STATE COMPARISON =====")
    print(
        "Raw exact tensors    :",
        f"{raw_comparison['exact_tensors']}/"
        f"{raw_comparison['total_tensors']}",
    )
    print(
        "Raw max abs diff     :",
        raw_comparison[
            "max_abs_difference"
        ],
    )
    print(
        "EMA exact tensors    :",
        f"{ema_comparison['exact_tensors']}/"
        f"{ema_comparison['total_tensors']}",
    )
    print(
        "EMA max abs diff     :",
        ema_comparison[
            "max_abs_difference"
        ],
    )
    print(
        "Optimizer differences:",
        len(optimizer_differences),
    )
    print(
        "Scaler differences   :",
        len(scaler_differences),
    )
    print(
        "Metadata differences :",
        metadata_differences,
    )

    print()
    print("===== FILE COMPARISON =====")
    print("FITLAB-01 bytes :", CHECKPOINT_01.stat().st_size)
    print("FITLAB-02 bytes :", CHECKPOINT_02.stat().st_size)
    print("FITLAB-01 SHA256:", sha256_01)
    print("FITLAB-02 SHA256:", sha256_02)
    print(
        "File bytes exact:",
        sha256_01 == sha256_02,
    )

    raw_exact = (
        raw_comparison["exact_tensors"]
        == raw_comparison["total_tensors"]
    )
    ema_exact = (
        ema_comparison["exact_tensors"]
        == ema_comparison["total_tensors"]
    )
    optimizer_exact = (
        len(optimizer_differences) == 0
    )
    scaler_exact = (
        len(scaler_differences) == 0
    )
    metadata_exact = (
        len(metadata_differences) == 0
    )

    assert raw_exact
    assert ema_exact
    assert optimizer_exact
    assert scaler_exact
    assert metadata_exact

    assert (
        raw_comparison[
            "max_abs_difference"
        ]
        == 0.0
    )
    assert (
        ema_comparison[
            "max_abs_difference"
        ]
        == 0.0
    )

    report = {
        "fitlab01_checkpoint": str(
            CHECKPOINT_01
        ),
        "fitlab02_checkpoint": str(
            CHECKPOINT_02
        ),
        "fitlab01_bytes": (
            CHECKPOINT_01.stat().st_size
        ),
        "fitlab02_bytes": (
            CHECKPOINT_02.stat().st_size
        ),
        "fitlab01_sha256": sha256_01,
        "fitlab02_sha256": sha256_02,
        "file_bytes_exact": (
            sha256_01 == sha256_02
        ),
        "raw_state": raw_comparison,
        "ema_state": ema_comparison,
        "optimizer_difference_count": len(
            optimizer_differences
        ),
        "optimizer_difference_paths": (
            optimizer_differences
        ),
        "scaler_difference_count": len(
            scaler_differences
        ),
        "metadata_differences": (
            metadata_differences
        ),
        "tensor_state_equivalent": (
            raw_exact
            and ema_exact
            and optimizer_exact
            and scaler_exact
            and metadata_exact
        ),
    }

    REPORT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    REPORT_PATH.write_text(
        json.dumps(report, indent=2)
        + "\n",
        encoding="utf-8",
    )

    print("Report          :", REPORT_PATH)
    print()
    print(
        "CROSS-SERVER B8 RESUME "
        "EQUIVALENCE: PASSED"
    )


if __name__ == "__main__":
    main()
