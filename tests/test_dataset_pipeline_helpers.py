from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))

from build_voc_corruptions import corrupt_image  # noqa: E402
from convert_exdark_to_yolo import parse_annotation as parse_exdark  # noqa: E402
from convert_visdrone_to_yolo import (  # noqa: E402
    parse_annotation as parse_visdrone,
)
from dataset_common import materialize_file, yolo_box  # noqa: E402


def test_materialize_file_copies_bytes(tmp_path: Path) -> None:
    source = tmp_path / "source.jpg"
    destination = tmp_path / "nested/destination.jpg"
    source.write_bytes(b"not-empty")

    materialize_file(source, destination)

    assert destination.read_bytes() == b"not-empty"
    assert not destination.is_symlink()


def test_yolo_box_clips_to_image() -> None:
    assert yolo_box(100, 50, -10, 5, 30, 60) == (
        0.1,
        0.55,
        0.2,
        0.9,
    )


def test_exdark_annotation_conversion(tmp_path: Path) -> None:
    annotation = tmp_path / "sample.jpg.txt"
    annotation.write_text(
        "% bbGt version=3\nTable 10 20 40 30 0 0 0 0 0 0 0\n",
        encoding="utf-8",
    )

    lines, invalid, unknown = parse_exdark(annotation, 100, 100)

    assert lines == ["11 0.30000000 0.35000000 0.40000000 0.30000000"]
    assert invalid == 0
    assert unknown == 0


def test_visdrone_ignores_region_and_maps_classes(tmp_path: Path) -> None:
    annotation = tmp_path / "sample.txt"
    annotation.write_text(
        "0,0,20,20,0,0,0,0\n10,20,40,30,1,4,0,0\n",
        encoding="utf-8",
    )

    lines, ignored, invalid = parse_visdrone(annotation, 100, 100)

    assert lines == ["3 0.30000000 0.35000000 0.40000000 0.30000000"]
    assert ignored == 1
    assert invalid == 0


def test_corruption_is_deterministic() -> None:
    source = Image.new("RGB", (8, 8), (100, 120, 140))
    first = corrupt_image(
        source,
        "gaussian_noise",
        3,
        np.random.default_rng(123),
    )
    second = corrupt_image(
        source,
        "gaussian_noise",
        3,
        np.random.default_rng(123),
    )

    assert np.array_equal(np.asarray(first), np.asarray(second))
