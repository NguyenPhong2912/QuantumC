from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any


IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
}


def materialize_file(source: Path, destination: Path) -> None:
    """Copy bytes and reject the empty-file failure seen on FITLAB NFS."""
    if not source.is_file():
        raise FileNotFoundError(f"Missing source file: {source}")

    source_size = source.stat().st_size

    if source_size <= 0:
        raise ValueError(f"Empty source file: {source}")

    destination.parent.mkdir(parents=True, exist_ok=True)

    if destination.exists() or destination.is_symlink():
        destination.unlink()

    shutil.copy2(source, destination)

    destination_size = destination.stat().st_size

    if destination_size != source_size:
        raise OSError(
            "Copied file size mismatch: "
            f"{source} ({source_size}) -> "
            f"{destination} ({destination_size})"
        )


def prepare_transaction(output_root: Path, force: bool) -> Path:
    temporary_root = output_root.with_name(
        f"{output_root.name}.tmp"
    )

    if output_root.exists() and not force:
        raise FileExistsError(
            f"{output_root} already exists. Use --force to rebuild it."
        )

    if temporary_root.exists():
        shutil.rmtree(temporary_root)

    temporary_root.mkdir(parents=True)
    return temporary_root


def commit_transaction(
    temporary_root: Path,
    output_root: Path,
) -> None:
    previous_root = output_root.with_name(
        f"{output_root.name}.previous"
    )

    if previous_root.exists():
        shutil.rmtree(previous_root)

    if output_root.exists():
        output_root.rename(previous_root)

    try:
        temporary_root.rename(output_root)
    except Exception:
        if previous_root.exists() and not output_root.exists():
            previous_root.rename(output_root)
        raise

    if previous_root.exists():
        shutil.rmtree(previous_root)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def yolo_box(
    image_width: int,
    image_height: int,
    x: float,
    y: float,
    width: float,
    height: float,
) -> tuple[float, float, float, float] | None:
    if image_width <= 0 or image_height <= 0:
        return None

    x1 = max(0.0, min(float(image_width), x))
    y1 = max(0.0, min(float(image_height), y))
    x2 = max(0.0, min(float(image_width), x + width))
    y2 = max(0.0, min(float(image_height), y + height))

    if x2 <= x1 or y2 <= y1:
        return None

    return (
        ((x1 + x2) / 2.0) / image_width,
        ((y1 + y2) / 2.0) / image_height,
        (x2 - x1) / image_width,
        (y2 - y1) / image_height,
    )


def format_yolo_line(
    class_id: int,
    box: tuple[float, float, float, float],
) -> str:
    return (
        f"{class_id} "
        f"{box[0]:.8f} {box[1]:.8f} "
        f"{box[2]:.8f} {box[3]:.8f}"
    )
