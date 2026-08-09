from __future__ import annotations

import argparse
import csv
import hashlib
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

from PIL import Image

from dataset_common import (
    IMAGE_EXTENSIONS,
    commit_transaction,
    format_yolo_line,
    materialize_file,
    prepare_transaction,
    write_json,
    yolo_box,
)


CLASS_NAMES = [
    "bicycle",
    "boat",
    "bottle",
    "bus",
    "car",
    "cat",
    "chair",
    "cup",
    "dog",
    "motorbike",
    "people",
    "table",
]
CLASS_TO_ID = {name: index for index, name in enumerate(CLASS_NAMES)}
EXPECTED_IMAGES = 7363


@dataclass(frozen=True)
class ExDarkStats:
    split: str
    images: int
    labels: int
    objects: int
    empty_labels: int
    invalid_boxes: int
    unknown_classes: int


def normalized_name(value: str) -> str:
    return value.strip().lower().replace(" ", "")


def parse_annotation(
    annotation_path: Path,
    width: int,
    height: int,
) -> tuple[list[str], int, int]:
    lines: list[str] = []
    invalid_boxes = 0
    unknown_classes = 0

    for raw_line in annotation_path.read_text(
        encoding="utf-8",
        errors="replace",
    ).splitlines():
        raw_line = raw_line.strip()
        if not raw_line or raw_line.startswith("%"):
            continue

        fields = raw_line.split()
        if len(fields) < 5:
            invalid_boxes += 1
            continue

        class_name = normalized_name(fields[0])
        if class_name not in CLASS_TO_ID:
            unknown_classes += 1
            continue

        try:
            x, y, box_width, box_height = map(float, fields[1:5])
        except ValueError:
            invalid_boxes += 1
            continue

        box = yolo_box(width, height, x, y, box_width, box_height)
        if box is None:
            invalid_boxes += 1
            continue

        lines.append(format_yolo_line(CLASS_TO_ID[class_name], box))

    return lines, invalid_boxes, unknown_classes


def assign_splits(
    images_by_category: dict[str, list[Path]],
    seed: int,
) -> dict[Path, str]:
    assignments: dict[Path, str] = {}

    for category, images in sorted(images_by_category.items()):
        ranked = sorted(
            images,
            key=lambda path: hashlib.sha256(
                f"{seed}:{category}:{path.name}".encode()
            ).hexdigest(),
        )
        train_end = int(len(ranked) * 0.8)
        val_end = train_end + int(len(ranked) * 0.1)

        for index, path in enumerate(ranked):
            if index < train_end:
                split = "train"
            elif index < val_end:
                split = "val"
            else:
                split = "test"
            assignments[path] = split

    return assignments


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    project_root = args.project_root.resolve()
    raw_root = project_root / "data/raw/exdark"
    image_root = raw_root / "ExDark"
    annotation_root = raw_root / "ExDark_Annno"
    output_root = project_root / "data/exdark"

    if not image_root.is_dir() or not annotation_root.is_dir():
        raise FileNotFoundError(
            f"Missing ExDark roots: {image_root}, {annotation_root}"
        )

    images_by_category: dict[str, list[Path]] = {}

    for category_dir in sorted(image_root.iterdir()):
        if not category_dir.is_dir():
            continue
        images_by_category[category_dir.name] = sorted(
            path
            for path in category_dir.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )

    total_images = sum(len(paths) for paths in images_by_category.values())
    if total_images != EXPECTED_IMAGES:
        raise RuntimeError(
            f"Expected {EXPECTED_IMAGES} ExDark images, found {total_images}"
        )

    assignments = assign_splits(images_by_category, args.seed)
    temporary_root = prepare_transaction(output_root, args.force)
    manifest_dir = temporary_root / "manifests"
    manifest_dir.mkdir(parents=True)
    stats_counters = {
        split: Counter()
        for split in ("train", "val", "test")
    }

    with (manifest_dir / "samples.csv").open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "split",
                "source_image",
                "source_annotation",
                "output_image",
                "objects",
            ]
        )

        processed = 0
        for category, images in sorted(images_by_category.items()):
            annotation_category = annotation_root / category
            category_slug = normalized_name(category)

            for source_image in images:
                split = assignments[source_image]
                source_annotation = (
                    annotation_category / f"{source_image.name}.txt"
                )
                if not source_annotation.is_file():
                    raise FileNotFoundError(
                        f"Missing ExDark annotation: {source_annotation}"
                    )

                with Image.open(source_image) as image:
                    width, height = image.size

                label_lines, invalid, unknown = parse_annotation(
                    source_annotation,
                    width,
                    height,
                )
                output_stem = f"{category_slug}_{source_image.stem}"
                destination_image = (
                    temporary_root
                    / "images"
                    / split
                    / f"{output_stem}{source_image.suffix.lower()}"
                )
                destination_label = (
                    temporary_root / "labels" / split / f"{output_stem}.txt"
                )
                materialize_file(source_image, destination_image)
                destination_label.parent.mkdir(parents=True, exist_ok=True)
                destination_label.write_text(
                    "\n".join(label_lines) + ("\n" if label_lines else ""),
                    encoding="utf-8",
                )

                counter = stats_counters[split]
                counter["images"] += 1
                counter["labels"] += 1
                counter["objects"] += len(label_lines)
                counter["empty_labels"] += int(not label_lines)
                counter["invalid_boxes"] += invalid
                counter["unknown_classes"] += unknown
                writer.writerow(
                    [
                        split,
                        source_image,
                        source_annotation,
                        destination_image.relative_to(temporary_root),
                        len(label_lines),
                    ]
                )
                processed += 1

                if processed % 1000 == 0:
                    print(f"ExDark: processed {processed:,} images", flush=True)

    stats = [
        ExDarkStats(split=split, **stats_counters[split])
        for split in ("train", "val", "test")
    ]
    write_json(
        manifest_dir / "conversion_stats.json",
        {
            "seed": args.seed,
            "split_policy": "per-class deterministic 80/10/10",
            "class_names": CLASS_NAMES,
            "splits": [asdict(item) for item in stats],
        },
    )

    if any(item.invalid_boxes or item.unknown_classes for item in stats):
        raise RuntimeError(f"ExDark conversion errors: {stats}")

    commit_transaction(temporary_root, output_root)
    print(f"EXDARK TO YOLO PASSED: {total_images:,} images")


if __name__ == "__main__":
    main()
