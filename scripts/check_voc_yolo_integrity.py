from __future__ import annotations

import csv
import json
import math
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

import yaml
from PIL import Image


IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
}


@dataclass(frozen=True)
class SplitCheck:
    split: str
    images: int
    labels: int
    paired: int
    broken_links: int
    unreadable_images: int
    malformed_lines: int
    invalid_class_ids: int
    invalid_coordinates: int
    nonfinite_values: int
    empty_labels: int
    objects: int
    min_class_id: int | None
    max_class_id: int | None


def image_files(path: Path) -> list[Path]:
    return sorted(
        item
        for item in path.iterdir()
        if item.suffix.lower() in IMAGE_EXTENSIONS
    )


def label_files(path: Path) -> list[Path]:
    return sorted(path.glob("*.txt"))


def check_split(
    *,
    split: str,
    image_dir: Path,
    label_dir: Path,
    expected_classes: int,
) -> tuple[SplitCheck, Counter[int]]:
    images = image_files(image_dir)
    labels = label_files(label_dir)

    image_stems = {
        image.stem
        for image in images
    }
    label_stems = {
        label.stem
        for label in labels
    }

    paired = len(image_stems & label_stems)

    broken_links = 0
    unreadable_images = 0
    malformed_lines = 0
    invalid_class_ids = 0
    invalid_coordinates = 0
    nonfinite_values = 0
    empty_labels = 0
    object_count = 0

    class_histogram: Counter[int] = Counter()

    for index, image_path in enumerate(images, start=1):
        if image_path.is_symlink() and not image_path.exists():
            broken_links += 1
            continue

        try:
            with Image.open(image_path) as image:
                image.verify()
        except Exception:
            unreadable_images += 1

        if index % 2000 == 0:
            print(
                f"{split}: verified {index:,} images",
                flush=True,
            )

    for index, label_path in enumerate(labels, start=1):
        lines = [
            line.strip()
            for line in label_path.read_text(
                encoding="utf-8"
            ).splitlines()
            if line.strip()
        ]

        if not lines:
            empty_labels += 1

        for line in lines:
            fields = line.split()

            if len(fields) != 5:
                malformed_lines += 1
                continue

            try:
                class_value = float(fields[0])
                coordinates = [
                    float(value)
                    for value in fields[1:]
                ]
            except ValueError:
                malformed_lines += 1
                continue

            all_values = [
                class_value,
                *coordinates,
            ]

            if not all(
                math.isfinite(value)
                for value in all_values
            ):
                nonfinite_values += 1
                continue

            class_id = int(class_value)

            if class_value != class_id:
                invalid_class_ids += 1
                continue

            if not 0 <= class_id < expected_classes:
                invalid_class_ids += 1
                continue

            x_center, y_center, width, height = coordinates

            if not (
                0.0 <= x_center <= 1.0
                and 0.0 <= y_center <= 1.0
                and 0.0 < width <= 1.0
                and 0.0 < height <= 1.0
            ):
                invalid_coordinates += 1
                continue

            object_count += 1
            class_histogram[class_id] += 1

        if index % 5000 == 0:
            print(
                f"{split}: checked {index:,} labels",
                flush=True,
            )

    class_ids = sorted(class_histogram)

    result = SplitCheck(
        split=split,
        images=len(images),
        labels=len(labels),
        paired=paired,
        broken_links=broken_links,
        unreadable_images=unreadable_images,
        malformed_lines=malformed_lines,
        invalid_class_ids=invalid_class_ids,
        invalid_coordinates=invalid_coordinates,
        nonfinite_values=nonfinite_values,
        empty_labels=empty_labels,
        objects=object_count,
        min_class_id=class_ids[0] if class_ids else None,
        max_class_id=class_ids[-1] if class_ids else None,
    )

    return result, class_histogram


def main() -> None:
    print("===== VOC YOLO INTEGRITY CHECK =====")

    project_root = Path(__file__).resolve().parents[1]
    dataset_root = project_root / "data/voc"
    yaml_path = project_root / "configs/datasets/voc.yaml"

    payload = yaml.safe_load(
        yaml_path.read_text(encoding="utf-8")
    )

    names = payload.get("names")

    if isinstance(names, dict):
        expected_classes = len(names)
    elif isinstance(names, list):
        expected_classes = len(names)
    else:
        raise RuntimeError(
            "VOC YAML does not contain a valid names map"
        )

    checks: list[SplitCheck] = []
    histograms: dict[str, dict[int, int]] = {}

    for split in ("train", "val"):
        check, histogram = check_split(
            split=split,
            image_dir=dataset_root / "images" / split,
            label_dir=dataset_root / "labels" / split,
            expected_classes=expected_classes,
        )

        checks.append(check)
        histograms[split] = dict(
            sorted(histogram.items())
        )

        print()
        print(
            f"{split:<5} | "
            f"images={check.images:,} | "
            f"labels={check.labels:,} | "
            f"paired={check.paired:,} | "
            f"objects={check.objects:,}"
        )
        print(
            f"      broken={check.broken_links} | "
            f"unreadable={check.unreadable_images} | "
            f"malformed={check.malformed_lines} | "
            f"bad_class={check.invalid_class_ids} | "
            f"bad_coords={check.invalid_coordinates} | "
            f"nonfinite={check.nonfinite_values} | "
            f"empty={check.empty_labels}"
        )

    train_stems = {
        path.stem
        for path in image_files(
            dataset_root / "images/train"
        )
    }
    val_stems = {
        path.stem
        for path in image_files(
            dataset_root / "images/val"
        )
    }

    overlap = train_stems & val_stems

    output_dir = dataset_root / "manifests"
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / "integrity_summary.csv"

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

    (
        output_dir / "class_histograms.json"
    ).write_text(
        json.dumps(
            histograms,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    train_check, val_check = checks

    assert expected_classes == 20

    assert train_check.images == 16551
    assert train_check.labels == 16551
    assert train_check.paired == 16551
    assert train_check.objects == 40058

    assert val_check.images == 4952
    assert val_check.labels == 4952
    assert val_check.paired == 4952
    assert val_check.objects == 12032

    for check in checks:
        assert check.broken_links == 0
        assert check.unreadable_images == 0
        assert check.malformed_lines == 0
        assert check.invalid_class_ids == 0
        assert check.invalid_coordinates == 0
        assert check.nonfinite_values == 0
        assert check.empty_labels == 0
        assert check.min_class_id == 0
        assert check.max_class_id == 19

    assert len(overlap) == 0

    print()
    print("Train/val overlap     :", len(overlap))
    print("Summary saved         :", csv_path)
    print("VOC YOLO INTEGRITY    : PASSED")


if __name__ == "__main__":
    main()
