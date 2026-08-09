from __future__ import annotations

import argparse
import csv
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
    "pedestrian",
    "people",
    "bicycle",
    "car",
    "van",
    "truck",
    "tricycle",
    "awning-tricycle",
    "bus",
    "motor",
]
RAW_SPLITS = {
    "train": "VisDrone2019-DET-train",
    "val": "VisDrone2019-DET-val",
    "test": "VisDrone2019-DET-test-dev",
}
EXPECTED_IMAGES = {"train": 6471, "val": 548, "test": 1610}


@dataclass(frozen=True)
class VisDroneStats:
    split: str
    images: int
    labels: int
    objects: int
    ignored_objects: int
    invalid_boxes: int
    empty_labels: int


def parse_annotation(
    annotation_path: Path,
    width: int,
    height: int,
) -> tuple[list[str], int, int]:
    if not annotation_path.is_file():
        return [], 0, 0

    lines: list[str] = []
    ignored = 0
    invalid = 0

    for raw_line in annotation_path.read_text(
        encoding="utf-8",
        errors="replace",
    ).splitlines():
        fields = [field.strip() for field in raw_line.split(",")]
        if len(fields) < 6:
            invalid += 1
            continue

        try:
            x, y, box_width, box_height = map(float, fields[:4])
            score = int(float(fields[4]))
            raw_class_id = int(float(fields[5]))
        except ValueError:
            invalid += 1
            continue

        if score <= 0 or not 1 <= raw_class_id <= 10:
            ignored += 1
            continue

        box = yolo_box(width, height, x, y, box_width, box_height)
        if box is None:
            invalid += 1
            continue

        lines.append(format_yolo_line(raw_class_id - 1, box))

    return lines, ignored, invalid


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    project_root = args.project_root.resolve()
    raw_root = project_root / "data/raw/visdrone2019"
    output_root = project_root / "data/visdrone2019"
    temporary_root = prepare_transaction(output_root, args.force)
    manifest_dir = temporary_root / "manifests"
    manifest_dir.mkdir(parents=True)
    stats: list[VisDroneStats] = []

    with (manifest_dir / "samples.csv").open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["split", "source_image", "source_annotation", "objects"]
        )

        for split, raw_name in RAW_SPLITS.items():
            split_root = raw_root / raw_name
            image_root = split_root / "images"
            annotation_root = split_root / "annotations"
            images = sorted(
                path
                for path in image_root.iterdir()
                if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
            )

            if len(images) != EXPECTED_IMAGES[split]:
                raise RuntimeError(
                    f"{split}: expected {EXPECTED_IMAGES[split]} images, "
                    f"found {len(images)}"
                )

            counter: Counter[str] = Counter()

            for index, source_image in enumerate(images, start=1):
                source_annotation = (
                    annotation_root / f"{source_image.stem}.txt"
                )
                with Image.open(source_image) as image:
                    width, height = image.size

                label_lines, ignored, invalid = parse_annotation(
                    source_annotation,
                    width,
                    height,
                )
                destination_image = (
                    temporary_root
                    / "images"
                    / split
                    / f"{source_image.stem}{source_image.suffix.lower()}"
                )
                destination_label = (
                    temporary_root
                    / "labels"
                    / split
                    / f"{source_image.stem}.txt"
                )
                materialize_file(source_image, destination_image)
                destination_label.parent.mkdir(parents=True, exist_ok=True)
                destination_label.write_text(
                    "\n".join(label_lines) + ("\n" if label_lines else ""),
                    encoding="utf-8",
                )
                counter["images"] += 1
                counter["labels"] += 1
                counter["objects"] += len(label_lines)
                counter["ignored_objects"] += ignored
                counter["invalid_boxes"] += invalid
                counter["empty_labels"] += int(not label_lines)
                writer.writerow(
                    [split, source_image, source_annotation, len(label_lines)]
                )

                if index % 1000 == 0:
                    print(
                        f"VisDrone {split}: processed {index:,} images",
                        flush=True,
                    )

            stats.append(VisDroneStats(split=split, **counter))

    write_json(
        manifest_dir / "conversion_stats.json",
        {
            "class_names": CLASS_NAMES,
            "splits": [asdict(item) for item in stats],
        },
    )

    if any(item.invalid_boxes for item in stats):
        raise RuntimeError(f"VisDrone invalid boxes detected: {stats}")

    commit_transaction(temporary_root, output_root)
    print("VISDRONE TO YOLO PASSED")


if __name__ == "__main__":
    main()
