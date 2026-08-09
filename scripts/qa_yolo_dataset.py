from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import subprocess
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import yaml
from PIL import Image, ImageDraw, ImageFont

from dataset_common import IMAGE_EXTENSIONS, write_json


@dataclass(frozen=True)
class SplitQa:
    split: str
    images: int
    labels: int
    paired: int
    missing_labels: int
    orphan_labels: int
    unreadable_images: int
    zero_byte_images: int
    malformed_lines: int
    invalid_class_ids: int
    invalid_coordinates: int
    nonfinite_values: int
    empty_labels: int
    objects: int


def class_names(payload: dict) -> list[str]:
    names = payload.get("names")

    if isinstance(names, list):
        return [str(name) for name in names]

    if isinstance(names, dict):
        return [
            str(names[index] if index in names else names[str(index)])
            for index in range(len(names))
        ]

    raise ValueError("Dataset YAML must contain a names list or map")


def resolve_dataset_root(yaml_path: Path, payload: dict) -> Path:
    root = Path(str(payload.get("path", "")))

    if not root.is_absolute():
        root = (yaml_path.parent / root).resolve()

    return root


def collect_images(path: Path) -> list[Path]:
    if not path.is_dir():
        return []

    return sorted(
        item
        for item in path.rglob("*")
        if item.is_file() and item.suffix.lower() in IMAGE_EXTENSIONS
    )


def collect_labels(path: Path) -> list[Path]:
    if not path.is_dir():
        return []
    return sorted(item for item in path.rglob("*.txt") if item.is_file())


def verify_label(
    label_path: Path,
    expected_classes: int,
    histogram: Counter[int],
) -> tuple[int, int, int, int, int, bool]:
    malformed = 0
    invalid_classes = 0
    invalid_coordinates = 0
    nonfinite = 0
    objects = 0
    lines = [
        line.strip()
        for line in label_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    for line in lines:
        fields = line.split()

        if len(fields) != 5:
            malformed += 1
            continue

        try:
            values = [float(value) for value in fields]
        except ValueError:
            malformed += 1
            continue

        if not all(math.isfinite(value) for value in values):
            nonfinite += 1
            continue

        class_value = values[0]
        class_id = int(class_value)

        if class_value != class_id or not 0 <= class_id < expected_classes:
            invalid_classes += 1
            continue

        x_center, y_center, width, height = values[1:]

        if not (
            0.0 <= x_center <= 1.0
            and 0.0 <= y_center <= 1.0
            and 0.0 < width <= 1.0
            and 0.0 < height <= 1.0
        ):
            invalid_coordinates += 1
            continue

        histogram[class_id] += 1
        objects += 1

    return (
        malformed,
        invalid_classes,
        invalid_coordinates,
        nonfinite,
        objects,
        not lines,
    )


def check_split(
    split: str,
    image_dir: Path,
    label_dir: Path,
    expected_classes: int,
    verify_images: bool,
) -> tuple[SplitQa, Counter[int]]:
    images = collect_images(image_dir)
    labels = collect_labels(label_dir)
    image_by_stem = {path.stem: path for path in images}
    label_by_stem = {path.stem: path for path in labels}
    paired_stems = image_by_stem.keys() & label_by_stem.keys()
    unreadable = 0
    zero_byte = 0
    histogram: Counter[int] = Counter()

    for index, image_path in enumerate(images, start=1):
        if image_path.stat().st_size == 0:
            zero_byte += 1
            continue

        if verify_images:
            try:
                with Image.open(image_path) as image:
                    image.verify()
            except Exception:
                unreadable += 1

        if index % 5000 == 0:
            print(f"{split}: verified {index:,} images", flush=True)

    malformed = 0
    invalid_classes = 0
    invalid_coordinates = 0
    nonfinite = 0
    empty = 0
    objects = 0

    for label_path in labels:
        result = verify_label(label_path, expected_classes, histogram)
        malformed += result[0]
        invalid_classes += result[1]
        invalid_coordinates += result[2]
        nonfinite += result[3]
        objects += result[4]
        empty += int(result[5])

    return (
        SplitQa(
            split=split,
            images=len(images),
            labels=len(labels),
            paired=len(paired_stems),
            missing_labels=len(image_by_stem.keys() - label_by_stem.keys()),
            orphan_labels=len(label_by_stem.keys() - image_by_stem.keys()),
            unreadable_images=unreadable,
            zero_byte_images=zero_byte,
            malformed_lines=malformed,
            invalid_class_ids=invalid_classes,
            invalid_coordinates=invalid_coordinates,
            nonfinite_values=nonfinite,
            empty_labels=empty,
            objects=objects,
        ),
        histogram,
    )


def draw_contact_sheet(
    image_dir: Path,
    label_dir: Path,
    names: list[str],
    output_path: Path,
    count: int,
    seed: int,
) -> None:
    images = collect_images(image_dir)
    ranked = sorted(
        images,
        key=lambda path: hashlib.sha256(
            f"{seed}:{path.name}".encode()
        ).hexdigest(),
    )[:count]

    if not ranked:
        return

    cell_width = 360
    cell_height = 240
    columns = 4
    rows = math.ceil(len(ranked) / columns)
    sheet = Image.new(
        "RGB",
        (columns * cell_width, rows * cell_height),
        "white",
    )
    font = ImageFont.load_default()

    for index, image_path in enumerate(ranked):
        with Image.open(image_path) as source:
            image = source.convert("RGB")

        draw = ImageDraw.Draw(image)
        label_path = label_dir / f"{image_path.stem}.txt"

        if label_path.is_file():
            for line in label_path.read_text(encoding="utf-8").splitlines():
                fields = line.split()
                if len(fields) != 5:
                    continue
                class_id = int(float(fields[0]))
                xc, yc, width, height = map(float, fields[1:])
                x1 = (xc - width / 2.0) * image.width
                y1 = (yc - height / 2.0) * image.height
                x2 = (xc + width / 2.0) * image.width
                y2 = (yc + height / 2.0) * image.height
                draw.rectangle((x1, y1, x2, y2), outline="red", width=2)
                draw.text(
                    (max(0, x1), max(0, y1 - 12)),
                    names[class_id],
                    fill="yellow",
                    font=font,
                    stroke_width=2,
                    stroke_fill="black",
                )

        image.thumbnail((cell_width, cell_height - 20))
        x_offset = (index % columns) * cell_width
        y_offset = (index // columns) * cell_height
        sheet.paste(image, (x_offset, y_offset + 20))
        ImageDraw.Draw(sheet).text(
            (x_offset + 4, y_offset + 4),
            image_path.name,
            fill="black",
            font=font,
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path, quality=92)


def git_commit(project_root: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=project_root,
            text=True,
        ).strip()
    except Exception:
        return "unknown"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-id", required=True)
    parser.add_argument("--yaml", type=Path)
    parser.add_argument("--splits", nargs="+", default=["train", "val"])
    parser.add_argument("--skip-image-verify", action="store_true")
    parser.add_argument("--contact-sheet-count", type=int, default=24)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    yaml_path = (
        args.yaml
        if args.yaml is not None
        else project_root / "configs/datasets" / f"{args.dataset_id}.yaml"
    ).resolve()
    payload = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    names = class_names(payload)
    dataset_root = resolve_dataset_root(yaml_path, payload)
    checks: list[SplitQa] = []
    histograms: dict[str, dict[int, int]] = {}
    split_stems: dict[str, set[str]] = {}

    for split in args.splits:
        relative_images = payload.get(split)
        if not isinstance(relative_images, str):
            raise ValueError(f"Missing {split} path in {yaml_path}")

        image_dir = dataset_root / relative_images
        label_relative = relative_images.replace("images", "labels", 1)
        label_dir = dataset_root / label_relative
        check, histogram = check_split(
            split,
            image_dir,
            label_dir,
            len(names),
            not args.skip_image_verify,
        )
        checks.append(check)
        histograms[split] = dict(sorted(histogram.items()))
        split_stems[split] = {path.stem for path in collect_images(image_dir)}
        print(asdict(check), flush=True)

    overlap = 0
    if "train" in split_stems and "val" in split_stems:
        overlap = len(split_stems["train"] & split_stems["val"])

    output_dir = project_root / "outputs/qa" / args.dataset_id
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_csv = output_dir / "integrity_summary.csv"

    with summary_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(checks[0])))
        writer.writeheader()
        for check in checks:
            writer.writerow(asdict(check))

    write_json(output_dir / "class_histograms.json", histograms)
    report_payload = {
        "dataset_id": args.dataset_id,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit(project_root),
        "yaml": str(yaml_path),
        "dataset_root": str(dataset_root),
        "classes": names,
        "splits": [asdict(check) for check in checks],
        "train_val_overlap": overlap,
    }
    write_json(output_dir / "dataset_report.json", report_payload)

    val_path = payload.get("val")
    if isinstance(val_path, str):
        draw_contact_sheet(
            dataset_root / val_path,
            dataset_root / val_path.replace("images", "labels", 1),
            names,
            output_dir / "contact_sheet.jpg",
            args.contact_sheet_count,
            args.seed,
        )

    report_lines = [
        f"# Dataset QA: {args.dataset_id}",
        "",
        f"- Generated (UTC): {report_payload['generated_at_utc']}",
        f"- Git commit: `{report_payload['git_commit']}`",
        f"- Classes: {len(names)}",
        f"- Train/val filename overlap: {overlap}",
        "",
        "| Split | Images | Labels | Paired | Objects | Empty | Errors |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]

    for check in checks:
        errors = sum(
            (
                check.missing_labels,
                check.orphan_labels,
                check.unreadable_images,
                check.zero_byte_images,
                check.malformed_lines,
                check.invalid_class_ids,
                check.invalid_coordinates,
                check.nonfinite_values,
            )
        )
        report_lines.append(
            f"| {check.split} | {check.images} | {check.labels} | "
            f"{check.paired} | {check.objects} | {check.empty_labels} | "
            f"{errors} |"
        )

    (output_dir / "dataset_report.md").write_text(
        "\n".join(report_lines) + "\n",
        encoding="utf-8",
    )

    for check in checks:
        assert check.images > 0
        assert check.images == check.labels == check.paired
        assert check.missing_labels == 0
        assert check.orphan_labels == 0
        assert check.unreadable_images == 0
        assert check.zero_byte_images == 0
        assert check.malformed_lines == 0
        assert check.invalid_class_ids == 0
        assert check.invalid_coordinates == 0
        assert check.nonfinite_values == 0

    assert overlap == 0
    print(f"YOLO DATASET QA PASSED: {args.dataset_id}")


if __name__ == "__main__":
    main()
