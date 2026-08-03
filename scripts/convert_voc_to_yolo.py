from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass
from pathlib import Path


CLASS_NAMES = [
    "aeroplane",
    "bicycle",
    "bird",
    "boat",
    "bottle",
    "bus",
    "car",
    "cat",
    "chair",
    "cow",
    "diningtable",
    "dog",
    "horse",
    "motorbike",
    "person",
    "pottedplant",
    "sheep",
    "sofa",
    "train",
    "tvmonitor",
]

CLASS_TO_ID = {
    name: index
    for index, name in enumerate(CLASS_NAMES)
}


@dataclass(frozen=True)
class ConversionStats:
    split: str
    images: int
    labels: int
    objects_written: int
    difficult_skipped: int
    empty_labels: int
    invalid_boxes: int
    unknown_classes: int


def read_split_ids(path: Path) -> list[str]:
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing split file: {path}"
        )

    image_ids = [
        line.strip()
        for line in path.read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip()
    ]

    if len(image_ids) != len(set(image_ids)):
        raise RuntimeError(
            f"Duplicate IDs in split file: {path}"
        )

    return image_ids


def parse_image_size(
    root: ET.Element,
    xml_path: Path,
) -> tuple[int, int]:
    size = root.find("size")

    if size is None:
        raise ValueError(
            f"Missing <size> in {xml_path}"
        )

    width_text = size.findtext("width")
    height_text = size.findtext("height")

    if width_text is None or height_text is None:
        raise ValueError(
            f"Missing image dimensions in {xml_path}"
        )

    width = int(float(width_text))
    height = int(float(height_text))

    if width <= 0 or height <= 0:
        raise ValueError(
            f"Invalid image dimensions in {xml_path}: "
            f"{width}x{height}"
        )

    return width, height


def convert_box(
    image_width: int,
    image_height: int,
    xmin: float,
    ymin: float,
    xmax: float,
    ymax: float,
) -> tuple[float, float, float, float] | None:
    # VOC coordinates are conventionally treated as 1-indexed.
    xmin = max(1.0, min(xmin, float(image_width)))
    xmax = max(1.0, min(xmax, float(image_width)))
    ymin = max(1.0, min(ymin, float(image_height)))
    ymax = max(1.0, min(ymax, float(image_height)))

    box_width = xmax - xmin
    box_height = ymax - ymin

    if box_width <= 0.0 or box_height <= 0.0:
        return None

    x_center = (
        ((xmin + xmax) / 2.0 - 1.0)
        / image_width
    )
    y_center = (
        ((ymin + ymax) / 2.0 - 1.0)
        / image_height
    )

    normalized_width = box_width / image_width
    normalized_height = box_height / image_height

    values = (
        x_center,
        y_center,
        normalized_width,
        normalized_height,
    )

    if not all(
        0.0 <= value <= 1.0
        for value in values
    ):
        return None

    return values


def convert_annotation(
    xml_path: Path,
) -> tuple[list[str], int, int, int]:
    root = ET.parse(xml_path).getroot()

    image_width, image_height = parse_image_size(
        root,
        xml_path,
    )

    lines: list[str] = []
    difficult_skipped = 0
    invalid_boxes = 0
    unknown_classes = 0

    for object_element in root.findall("object"):
        class_name = (
            object_element.findtext("name", "")
            .strip()
        )

        if class_name not in CLASS_TO_ID:
            unknown_classes += 1
            continue

        difficult_text = object_element.findtext(
            "difficult",
            "0",
        )

        try:
            difficult = int(difficult_text)
        except ValueError:
            difficult = 0

        if difficult == 1:
            difficult_skipped += 1
            continue

        box = object_element.find("bndbox")

        if box is None:
            invalid_boxes += 1
            continue

        try:
            xmin = float(box.findtext("xmin", "nan"))
            ymin = float(box.findtext("ymin", "nan"))
            xmax = float(box.findtext("xmax", "nan"))
            ymax = float(box.findtext("ymax", "nan"))
        except ValueError:
            invalid_boxes += 1
            continue

        converted = convert_box(
            image_width=image_width,
            image_height=image_height,
            xmin=xmin,
            ymin=ymin,
            xmax=xmax,
            ymax=ymax,
        )

        if converted is None:
            invalid_boxes += 1
            continue

        class_id = CLASS_TO_ID[class_name]

        lines.append(
            f"{class_id} "
            f"{converted[0]:.8f} "
            f"{converted[1]:.8f} "
            f"{converted[2]:.8f} "
            f"{converted[3]:.8f}"
        )

    return (
        lines,
        difficult_skipped,
        invalid_boxes,
        unknown_classes,
    )


def create_relative_symlink(
    source: Path,
    destination: Path,
) -> None:
    if not source.is_file():
        raise FileNotFoundError(
            f"Missing image: {source}"
        )

    if destination.exists() or destination.is_symlink():
        destination.unlink()

    relative_target = os.path.relpath(
        source.resolve(),
        destination.parent.resolve(),
    )

    destination.symlink_to(relative_target)


def process_split(
    *,
    split_name: str,
    sources: list[tuple[str, Path, Path]],
    image_output: Path,
    label_output: Path,
    manifest_output: Path,
) -> ConversionStats:
    image_output.mkdir(
        parents=True,
        exist_ok=True,
    )
    label_output.mkdir(
        parents=True,
        exist_ok=True,
    )

    image_count = 0
    label_count = 0
    object_count = 0
    difficult_skipped = 0
    empty_labels = 0
    invalid_boxes = 0
    unknown_classes = 0

    with manifest_output.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as manifest_handle:
        writer = csv.writer(manifest_handle)

        writer.writerow(
            [
                "split",
                "year",
                "image_id",
                "output_stem",
                "source_image",
                "source_xml",
                "object_count",
            ]
        )

        for year, voc_root, split_file in sources:
            image_ids = read_split_ids(split_file)

            for image_id in image_ids:
                output_stem = f"{year}_{image_id}"

                source_image = (
                    voc_root
                    / "JPEGImages"
                    / f"{image_id}.jpg"
                )

                source_xml = (
                    voc_root
                    / "Annotations"
                    / f"{image_id}.xml"
                )

                destination_image = (
                    image_output
                    / f"{output_stem}.jpg"
                )

                destination_label = (
                    label_output
                    / f"{output_stem}.txt"
                )

                if not source_xml.is_file():
                    raise FileNotFoundError(
                        f"Missing annotation: {source_xml}"
                    )

                (
                    label_lines,
                    skipped,
                    invalid,
                    unknown,
                ) = convert_annotation(source_xml)

                create_relative_symlink(
                    source=source_image,
                    destination=destination_image,
                )

                destination_label.write_text(
                    (
                        "\n".join(label_lines) + "\n"
                        if label_lines
                        else ""
                    ),
                    encoding="utf-8",
                )

                writer.writerow(
                    [
                        split_name,
                        year,
                        image_id,
                        output_stem,
                        str(source_image),
                        str(source_xml),
                        len(label_lines),
                    ]
                )

                image_count += 1
                label_count += 1
                object_count += len(label_lines)
                difficult_skipped += skipped
                invalid_boxes += invalid
                unknown_classes += unknown

                if not label_lines:
                    empty_labels += 1

                if image_count % 1000 == 0:
                    print(
                        f"{split_name}: "
                        f"{image_count:,} images processed",
                        flush=True,
                    )

    return ConversionStats(
        split=split_name,
        images=image_count,
        labels=label_count,
        objects_written=object_count,
        difficult_skipped=difficult_skipped,
        empty_labels=empty_labels,
        invalid_boxes=invalid_boxes,
        unknown_classes=unknown_classes,
    )


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
    )

    parser.add_argument(
        "--force",
        action="store_true",
    )

    args = parser.parse_args()

    project_root = args.project_root.resolve()

    raw_root = (
        project_root
        / "data/raw/voc/extracted/VOCdevkit"
    )

    output_root = project_root / "data/voc"
    temporary_root = project_root / "data/voc.tmp"

    voc2007 = raw_root / "VOC2007"
    voc2012 = raw_root / "VOC2012"

    required_directories = [
        voc2007 / "JPEGImages",
        voc2007 / "Annotations",
        voc2007 / "ImageSets/Main",
        voc2012 / "JPEGImages",
        voc2012 / "Annotations",
        voc2012 / "ImageSets/Main",
    ]

    for directory in required_directories:
        if not directory.is_dir():
            raise FileNotFoundError(
                f"Missing required directory: {directory}"
            )

    if output_root.exists():
        if not args.force:
            raise FileExistsError(
                f"{output_root} already exists. "
                "Use --force to rebuild it."
            )

        print(
            f"Removing existing output: {output_root}"
        )
        shutil.rmtree(output_root)

    if temporary_root.exists():
        print(
            "Removing incomplete temporary output: "
            f"{temporary_root}"
        )
        shutil.rmtree(temporary_root)

    manifests = temporary_root / "manifests"
    manifests.mkdir(
        parents=True,
        exist_ok=True,
    )

    train_stats = process_split(
        split_name="train",
        sources=[
            (
                "2007",
                voc2007,
                voc2007
                / "ImageSets/Main/trainval.txt",
            ),
            (
                "2012",
                voc2012,
                voc2012
                / "ImageSets/Main/trainval.txt",
            ),
        ],
        image_output=temporary_root
        / "images/train",
        label_output=temporary_root
        / "labels/train",
        manifest_output=manifests
        / "train.csv",
    )

    val_stats = process_split(
        split_name="val",
        sources=[
            (
                "2007",
                voc2007,
                voc2007
                / "ImageSets/Main/test.txt",
            ),
        ],
        image_output=temporary_root
        / "images/val",
        label_output=temporary_root
        / "labels/val",
        manifest_output=manifests
        / "val.csv",
    )

    stats_payload = {
        "class_names": CLASS_NAMES,
        "train": asdict(train_stats),
        "val": asdict(val_stats),
    }

    (
        manifests / "conversion_stats.json"
    ).write_text(
        json.dumps(
            stats_payload,
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    assert train_stats.images == 16551
    assert train_stats.labels == 16551
    assert val_stats.images == 4952
    assert val_stats.labels == 4952

    if train_stats.invalid_boxes != 0:
        raise RuntimeError(
            "Invalid train boxes detected: "
            f"{train_stats.invalid_boxes}"
        )

    if val_stats.invalid_boxes != 0:
        raise RuntimeError(
            "Invalid validation boxes detected: "
            f"{val_stats.invalid_boxes}"
        )

    if train_stats.unknown_classes != 0:
        raise RuntimeError(
            "Unknown train classes detected: "
            f"{train_stats.unknown_classes}"
        )

    if val_stats.unknown_classes != 0:
        raise RuntimeError(
            "Unknown validation classes detected: "
            f"{val_stats.unknown_classes}"
        )

    temporary_root.rename(output_root)

    print()
    print("===== VOC TO YOLO CONVERSION =====")

    for stats in (train_stats, val_stats):
        print(
            f"{stats.split:<5} | "
            f"images={stats.images:,} | "
            f"labels={stats.labels:,} | "
            f"objects={stats.objects_written:,} | "
            f"difficult_skipped="
            f"{stats.difficult_skipped:,} | "
            f"empty={stats.empty_labels:,}"
        )

    print("Output:", output_root)
    print("VOC TO YOLO CONVERSION: PASSED")


if __name__ == "__main__":
    main()
