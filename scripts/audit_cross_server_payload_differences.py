from __future__ import annotations

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
    "cross_server_b8_full_payload_differences.json"
)


def summarize_value(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return {
            "type": "tensor",
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "device": str(value.device),
        }

    if isinstance(value, Path):
        return str(value)

    if isinstance(value, dict):
        return {
            str(key): summarize_value(item)
            for key, item in value.items()
        }

    if isinstance(value, tuple):
        return [
            summarize_value(item)
            for item in value
        ]

    if isinstance(value, list):
        return [
            summarize_value(item)
            for item in value
        ]

    if isinstance(
        value,
        (str, int, float, bool),
    ) or value is None:
        return value

    return repr(value)


def compare_complete_tree(
    first: Any,
    second: Any,
    *,
    path: str = "checkpoint",
) -> list[dict[str, Any]]:
    differences: list[dict[str, Any]] = []

    if isinstance(first, torch.Tensor):
        if not isinstance(second, torch.Tensor):
            differences.append(
                {
                    "path": path,
                    "reason": "type",
                    "fitlab01": summarize_value(first),
                    "fitlab02": summarize_value(second),
                }
            )
            return differences

        if first.dtype != second.dtype:
            differences.append(
                {
                    "path": path,
                    "reason": "tensor_dtype",
                    "fitlab01": str(first.dtype),
                    "fitlab02": str(second.dtype),
                }
            )

        if first.shape != second.shape:
            differences.append(
                {
                    "path": path,
                    "reason": "tensor_shape",
                    "fitlab01": list(first.shape),
                    "fitlab02": list(second.shape),
                }
            )
            return differences

        if not torch.equal(first, second):
            record: dict[str, Any] = {
                "path": path,
                "reason": "tensor_value",
                "dtype": str(first.dtype),
                "shape": list(first.shape),
            }

            if (
                first.is_floating_point()
                and second.is_floating_point()
                and first.numel() > 0
            ):
                record["max_abs_difference"] = float(
                    (
                        first.to(torch.float64)
                        - second.to(torch.float64)
                    )
                    .abs()
                    .max()
                )

            differences.append(record)

        return differences

    if type(first) is not type(second):
        differences.append(
            {
                "path": path,
                "reason": "type",
                "fitlab01": {
                    "type": type(first).__name__,
                    "value": summarize_value(first),
                },
                "fitlab02": {
                    "type": type(second).__name__,
                    "value": summarize_value(second),
                },
            }
        )
        return differences

    if isinstance(first, dict):
        keys_01 = set(first)
        keys_02 = set(second)

        for key in sorted(
            keys_01.difference(keys_02),
            key=str,
        ):
            differences.append(
                {
                    "path": f"{path}.{key}",
                    "reason": "missing_from_fitlab02",
                    "fitlab01": summarize_value(
                        first[key]
                    ),
                }
            )

        for key in sorted(
            keys_02.difference(keys_01),
            key=str,
        ):
            differences.append(
                {
                    "path": f"{path}.{key}",
                    "reason": "missing_from_fitlab01",
                    "fitlab02": summarize_value(
                        second[key]
                    ),
                }
            )

        for key in sorted(
            keys_01.intersection(keys_02),
            key=str,
        ):
            differences.extend(
                compare_complete_tree(
                    first[key],
                    second[key],
                    path=f"{path}.{key}",
                )
            )

        return differences

    if isinstance(first, (list, tuple)):
        if len(first) != len(second):
            differences.append(
                {
                    "path": path,
                    "reason": "sequence_length",
                    "fitlab01": len(first),
                    "fitlab02": len(second),
                }
            )

        for index, (item_01, item_02) in enumerate(
            zip(first, second)
        ):
            differences.extend(
                compare_complete_tree(
                    item_01,
                    item_02,
                    path=f"{path}[{index}]",
                )
            )

        return differences

    if first != second:
        differences.append(
            {
                "path": path,
                "reason": "value",
                "fitlab01": summarize_value(first),
                "fitlab02": summarize_value(second),
            }
        )

    return differences


def main() -> None:
    print(
        "===== CROSS-SERVER FULL PAYLOAD "
        "DIFFERENCE AUDIT ====="
    )

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

    differences = compare_complete_tree(
        payload_01,
        payload_02,
    )

    tensor_differences = [
        item
        for item in differences
        if item["reason"].startswith("tensor")
    ]

    non_tensor_differences = [
        item
        for item in differences
        if not item["reason"].startswith("tensor")
    ]

    print("FITLAB-01:", CHECKPOINT_01)
    print("FITLAB-02:", CHECKPOINT_02)
    print()
    print("Total differences     :", len(differences))
    print("Tensor differences    :", len(tensor_differences))
    print(
        "Non-tensor differences:",
        len(non_tensor_differences),
    )

    print()
    print("===== DIFFERENCE PATHS =====")

    for item in differences:
        print()
        print("Path   :", item["path"])
        print("Reason :", item["reason"])

        if "fitlab01" in item:
            print(
                "FITLAB-01:",
                item["fitlab01"],
            )

        if "fitlab02" in item:
            print(
                "FITLAB-02:",
                item["fitlab02"],
            )

        if "max_abs_difference" in item:
            print(
                "Max diff:",
                item["max_abs_difference"],
            )

    assert len(tensor_differences) == 0

    report = {
        "fitlab01_checkpoint": str(
            CHECKPOINT_01
        ),
        "fitlab02_checkpoint": str(
            CHECKPOINT_02
        ),
        "total_difference_count": len(
            differences
        ),
        "tensor_difference_count": len(
            tensor_differences
        ),
        "non_tensor_difference_count": len(
            non_tensor_differences
        ),
        "differences": differences,
    }

    REPORT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    REPORT_PATH.write_text(
        json.dumps(
            report,
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    print()
    print("Report:", REPORT_PATH)
    print()
    print(
        "CROSS-SERVER FULL PAYLOAD AUDIT: PASSED"
    )


if __name__ == "__main__":
    main()
