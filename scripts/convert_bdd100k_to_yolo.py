from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path

from PIL import Image

from dataset_common import (
    commit_transaction,
    format_yolo_line,
    materialize_file,
    prepare_transaction,
    write_json,
    yolo_box,
)


CLASS_NAMES = [
    "pedestrian",
    "rider",
    "car",
    "truck",
    "bus",
    "train",
    "motorcycle",
    "bicycle",
    "traffic light",
    "traffic sign",
]
RAW_CLASS_TO_ID = {
    "person": 0,
    "pedestrian": 0,
    "rider": 1,
    "car": 2,
    "truck": 3,
    "bus": 4,
    "train": 5,
    "motor": 6,
    "motorcycle": 6,
    "bike": 7,
    "bicycle": 7,
    "traffic light": 8,
    "traffic sign": 9,
}
EXPECTED_IMAGES = {"train": 70000, "val": 10000}


@dataclass(frozen=True)
class BddStats:
    split: str
    images: int
    labels: int
    objects: int
    empty_labels: int
    ignored_categories: int
    invalid_boxes: int


def find_one(root: Path, patterns: list[str]) -> Path:
    matches: list[Path] = []
    for pattern in patterns:
        matches.extend(root.rglob(pattern))
    unique = sorted(set(matches))
    if len(unique) != 1:
        raise RuntimeError(
            f"Expected exactly one match for {patterns}, found {unique}"
        )
    return unique[0]


def find_image_root(raw_root: Path, split: str) -> Path:
    matches = sorted(
        path
        for path in raw_root.rglob(split)
        if path.is_dir() and path.parent.name == "100k"
    )
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected one BDD100K 100k/{split} directory, found {matches}"
        )
    return matches[0]


def convert_split(
    split: str,
    raw_root: Path,
    temporary_root: Path,
    workers: int,
) -> BddStats:
    annotation_path = find_one(
        raw_root,
        [
            f"bdd100k_labels_images_{split}.json",
            f"det_{split}.json",
        ],
    )
    image_root = find_image_root(raw_root, split)
    frames = json.loads(annotation_path.read_text(encoding="utf-8"))

    if len(frames) != EXPECTED_IMAGES[split]:
        raise RuntimeError(
            f"{split}: expected {EXPECTED_IMAGES[split]} frames, "
            f"found {len(frames)}"
        )

    image_output = temporary_root / "images" / split
    label_output = temporary_root / "labels" / split
    image_output.mkdir(parents=True)
    label_output.mkdir(parents=True)
    counter: Counter[str] = Counter(
        {
            "images": 0,
            "labels": 0,
            "objects": 0,
            "empty_labels": 0,
            "ignored_categories": 0,
            "invalid_boxes": 0,
        }
    )
    rows: list[list[object]] = []
    copy_jobs: list[tuple[Path, Path]] = []

    for index, frame in enumerate(frames, start=1):
        file_name = str(frame["name"])
        source_image = image_root / file_name
        destination_image = image_output / file_name
        destination_label = label_output / f"{Path(file_name).stem}.txt"

        with Image.open(source_image) as image:
            width, height = image.size

        label_lines: list[str] = []
        for label in frame.get("labels", []):
            category = str(label.get("category", "")).strip().lower()
            box2d = label.get("box2d")

            if category not in RAW_CLASS_TO_ID or not isinstance(box2d, dict):
                counter["ignored_categories"] += 1
                continue

            try:
                x1 = float(box2d["x1"])
                y1 = float(box2d["y1"])
                x2 = float(box2d["x2"])
                y2 = float(box2d["y2"])
            except (KeyError, TypeError, ValueError):
                counter["invalid_boxes"] += 1
                continue

            box = yolo_box(width, height, x1, y1, x2 - x1, y2 - y1)
            if box is None:
                counter["invalid_boxes"] += 1
                continue

            label_lines.append(format_yolo_line(RAW_CLASS_TO_ID[category], box))

        destination_label.write_text(
            "\n".join(label_lines) + ("\n" if label_lines else ""),
            encoding="utf-8",
        )
        copy_jobs.append((source_image, destination_image))
        counter["images"] += 1
        counter["labels"] += 1
        counter["objects"] += len(label_lines)
        counter["empty_labels"] += int(not label_lines)
        rows.append([split, file_name, len(label_lines)])

        if index % 10000 == 0:
            print(f"BDD100K {split}: prepared {index:,} labels", flush=True)

    def copy_job(job: tuple[Path, Path]) -> None:
        materialize_file(job[0], job[1])

    with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
        for index, _ in enumerate(executor.map(copy_job, copy_jobs), start=1):
            if index % 10000 == 0:
                print(f"BDD100K {split}: copied {index:,} images", flush=True)

    with (temporary_root / "manifests" / f"{split}.csv").open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(["split", "file_name", "objects"])
        writer.writerows(rows)

    return BddStats(split=split, **counter)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
    )
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    project_root = args.project_root.resolve()
    raw_root = project_root / "data/raw/bdd100k"
    output_root = project_root / "data/bdd100k"
    temporary_root = prepare_transaction(output_root, args.force)
    (temporary_root / "manifests").mkdir(parents=True)
    stats = [
        convert_split(split, raw_root, temporary_root, args.workers)
        for split in ("train", "val")
    ]
    write_json(
        temporary_root / "manifests/conversion_stats.json",
        {"class_names": CLASS_NAMES, "splits": [asdict(x) for x in stats]},
    )

    if any(item.invalid_boxes for item in stats):
        raise RuntimeError(f"BDD100K invalid boxes detected: {stats}")

    commit_transaction(temporary_root, output_root)
    print("BDD100K TO YOLO PASSED")


if __name__ == "__main__":
    main()
