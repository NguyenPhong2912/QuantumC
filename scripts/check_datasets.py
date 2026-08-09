from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
from pathlib import Path

import yaml

from src.qvisionframe.dataset_registry import (
    dataset_status,
    registry,
    required_paths,
)


IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
}


@dataclass(frozen=True)
class DatasetCheck:
    dataset_id: str
    name: str
    status: str
    yaml_exists: bool
    yaml_valid: bool
    declared_classes: int
    expected_classes: int | None
    class_count_matches: bool
    train_images_exist: bool
    val_images_exist: bool
    train_labels_exist: bool
    val_labels_exist: bool
    train_image_count: int
    val_image_count: int
    train_label_count: int
    val_label_count: int
    paired_train_count: int
    paired_val_count: int


def count_images(path: Path) -> int:
    if not path.exists():
        return 0

    return sum(
        1
        for item in path.rglob("*")
        if item.is_file()
        and item.suffix.lower() in IMAGE_EXTENSIONS
    )


def count_labels(path: Path) -> int:
    if not path.exists():
        return 0

    return sum(
        1
        for item in path.rglob("*.txt")
        if item.is_file()
    )


def image_stems(path: Path) -> set[str]:
    if not path.exists():
        return set()

    return {
        item.stem
        for item in path.rglob("*")
        if item.is_file()
        and item.suffix.lower() in IMAGE_EXTENSIONS
    }


def label_stems(path: Path) -> set[str]:
    if not path.exists():
        return set()

    return {
        item.stem
        for item in path.rglob("*.txt")
        if item.is_file()
    }


def read_yaml_class_count(path: Path) -> tuple[bool, int]:
    if not path.exists():
        return False, 0

    try:
        payload = yaml.safe_load(
            path.read_text(encoding="utf-8")
        )

        names = payload.get("names")

        if isinstance(names, dict):
            return True, len(names)

        if isinstance(names, list):
            return True, len(names)

        return False, 0

    except Exception:
        return False, 0


def main() -> None:
    print("===== DATASET PREFLIGHT CHECK =====")

    checks: list[DatasetCheck] = []

    for dataset_id, spec in registry().items():
        yaml_valid, declared_classes = (
            read_yaml_class_count(spec.yaml_path)
        )

        train_images = count_images(spec.train_images)
        val_images = count_images(spec.val_images)
        train_labels = count_labels(spec.train_labels)
        val_labels = count_labels(spec.val_labels)

        paired_train = len(
            image_stems(spec.train_images)
            & label_stems(spec.train_labels)
        )
        paired_val = len(
            image_stems(spec.val_images)
            & label_stems(spec.val_labels)
        )

        class_count_matches = (
            spec.expected_classes is None
            or declared_classes == spec.expected_classes
        )

        status = dataset_status(
            spec,
            declared_classes=declared_classes,
            train_image_count=train_images,
            val_image_count=val_images,
            train_label_count=train_labels,
            val_label_count=val_labels,
        )

        if status == "ready" and not (
            train_images == train_labels == paired_train
            and val_images == val_labels == paired_val
        ):
            status = "partial"

        check = DatasetCheck(
            dataset_id=dataset_id,
            name=spec.name,
            status=status,
            yaml_exists=spec.yaml_path.exists(),
            yaml_valid=yaml_valid,
            declared_classes=declared_classes,
            expected_classes=spec.expected_classes,
            class_count_matches=class_count_matches,
            train_images_exist=spec.train_images.exists(),
            val_images_exist=spec.val_images.exists(),
            train_labels_exist=spec.train_labels.exists(),
            val_labels_exist=spec.val_labels.exists(),
            train_image_count=train_images,
            val_image_count=val_images,
            train_label_count=train_labels,
            val_label_count=val_labels,
            paired_train_count=paired_train,
            paired_val_count=paired_val,
        )

        checks.append(check)

        print(
            f"{dataset_id:<14} | "
            f"status={check.status:<7} | "
            f"classes={declared_classes}/"
            f"{spec.expected_classes} | "
            f"train={train_images}/{train_labels} | "
            f"val={val_images}/{val_labels}"
        )

        for path in required_paths(spec):
            print(
                "  ",
                "OK     " if path.exists() else "MISSING",
                path,
            )

    output_dir = Path("outputs/inventory")
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / "dataset_preflight.csv"

    with csv_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(asdict(checks[0]).keys()),
        )
        writer.writeheader()

        for check in checks:
            writer.writerow(asdict(check))

    assert len(checks) == 5
    assert all(check.yaml_exists for check in checks)
    assert all(check.yaml_valid for check in checks)
    assert all(
        check.class_count_matches
        for check in checks
    ), "One or more dataset class maps are incomplete"

    print("-" * 88)
    print("Preflight saved      :", csv_path)
    print(
        "Ready datasets       :",
        sum(check.status == "ready" for check in checks),
    )
    print(
        "Partial datasets     :",
        sum(check.status == "partial" for check in checks),
    )
    print(
        "Missing datasets     :",
        sum(check.status == "missing" for check in checks),
    )
    print("DATASET PREFLIGHT    : PASSED")


if __name__ == "__main__":
    main()
