from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path

from dataset_common import (
    commit_transaction,
    format_yolo_line,
    materialize_file,
    prepare_transaction,
    write_json,
    yolo_box,
)


EXPECTED_IMAGES = {"train2017": 118287, "val2017": 5000}


@dataclass(frozen=True)
class CocoStats:
    split: str
    images: int
    labels: int
    objects: int
    empty_labels: int
    crowd_skipped: int
    invalid_boxes: int


def convert_split(
    split: str,
    raw_root: Path,
    temporary_root: Path,
    workers: int,
) -> CocoStats:
    annotation_path = (
        raw_root / "annotations" / f"instances_{split}.json"
    )
    image_root = raw_root / split
    payload = json.loads(annotation_path.read_text(encoding="utf-8"))
    categories = sorted(payload["categories"], key=lambda item: item["id"])
    category_to_class = {
        int(category["id"]): index
        for index, category in enumerate(categories)
    }
    images = sorted(payload["images"], key=lambda item: int(item["id"]))

    if len(images) != EXPECTED_IMAGES[split]:
        raise RuntimeError(
            f"{split}: expected {EXPECTED_IMAGES[split]} images, "
            f"found {len(images)}"
        )

    annotations_by_image: dict[int, list[dict]] = defaultdict(list)
    for annotation in payload["annotations"]:
        annotations_by_image[int(annotation["image_id"])].append(annotation)

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
            "crowd_skipped": 0,
            "invalid_boxes": 0,
        }
    )
    manifest_rows: list[list[object]] = []
    copy_jobs: list[tuple[Path, Path]] = []

    for index, image in enumerate(images, start=1):
        image_id = int(image["id"])
        file_name = str(image["file_name"])
        width = int(image["width"])
        height = int(image["height"])
        source_image = image_root / file_name
        destination_image = image_output / file_name
        destination_label = label_output / f"{Path(file_name).stem}.txt"
        label_lines: list[str] = []

        for annotation in annotations_by_image.get(image_id, []):
            if int(annotation.get("iscrowd", 0)) == 1:
                counter["crowd_skipped"] += 1
                continue

            raw_bbox = annotation.get("bbox")
            if not isinstance(raw_bbox, list) or len(raw_bbox) != 4:
                counter["invalid_boxes"] += 1
                continue

            box = yolo_box(
                width,
                height,
                float(raw_bbox[0]),
                float(raw_bbox[1]),
                float(raw_bbox[2]),
                float(raw_bbox[3]),
            )
            class_id = category_to_class.get(int(annotation["category_id"]))

            if box is None or class_id is None:
                counter["invalid_boxes"] += 1
                continue

            label_lines.append(format_yolo_line(class_id, box))

        destination_label.write_text(
            "\n".join(label_lines) + ("\n" if label_lines else ""),
            encoding="utf-8",
        )
        copy_jobs.append((source_image, destination_image))
        manifest_rows.append(
            [split, image_id, file_name, len(label_lines)]
        )
        counter["images"] += 1
        counter["labels"] += 1
        counter["objects"] += len(label_lines)
        counter["empty_labels"] += int(not label_lines)

        if index % 10000 == 0:
            print(
                f"COCO {split}: prepared {index:,} labels",
                flush=True,
            )

    def copy_job(job: tuple[Path, Path]) -> None:
        materialize_file(job[0], job[1])

    with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
        for index, _ in enumerate(executor.map(copy_job, copy_jobs), start=1):
            if index % 10000 == 0:
                print(
                    f"COCO {split}: copied {index:,} images",
                    flush=True,
                )

    with (temporary_root / "manifests" / f"{split}.csv").open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(["split", "image_id", "file_name", "objects"])
        writer.writerows(manifest_rows)

    return CocoStats(split=split, **counter)


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
    raw_root = project_root / "data/raw/coco2017"
    output_root = project_root / "data/coco2017"
    temporary_root = prepare_transaction(output_root, args.force)
    (temporary_root / "manifests").mkdir(parents=True)
    stats = [
        convert_split(
            split,
            raw_root,
            temporary_root,
            args.workers,
        )
        for split in ("train2017", "val2017")
    ]
    write_json(
        temporary_root / "manifests/conversion_stats.json",
        {"splits": [asdict(item) for item in stats]},
    )

    if any(item.invalid_boxes for item in stats):
        raise RuntimeError(f"COCO invalid boxes detected: {stats}")

    commit_transaction(temporary_root, output_root)
    print("COCO2017 TO YOLO PASSED")


if __name__ == "__main__":
    main()
