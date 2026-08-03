from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal


DatasetStatus = Literal[
    "missing",
    "partial",
    "ready",
]


@dataclass(frozen=True)
class DatasetSpec:
    dataset_id: str
    name: str
    task: str
    root: Path
    yaml_path: Path
    train_images: Path
    val_images: Path
    test_images: Path | None
    train_labels: Path
    val_labels: Path
    test_labels: Path | None
    expected_classes: int | None
    notes: str


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def data_root() -> Path:
    return project_root() / "data"


def registry() -> dict[str, DatasetSpec]:
    root = data_root()

    return {
        "coco2017": DatasetSpec(
            dataset_id="coco2017",
            name="COCO 2017",
            task="general object detection",
            root=root / "coco2017",
            yaml_path=project_root()
            / "configs/datasets/coco2017.yaml",
            train_images=root / "coco2017/images/train2017",
            val_images=root / "coco2017/images/val2017",
            test_images=root / "coco2017/images/test2017",
            train_labels=root / "coco2017/labels/train2017",
            val_labels=root / "coco2017/labels/val2017",
            test_labels=None,
            expected_classes=80,
            notes=(
                "Primary general benchmark. Use official 2017 split."
            ),
        ),
        "voc": DatasetSpec(
            dataset_id="voc",
            name="Pascal VOC",
            task="general object detection",
            root=root / "voc",
            yaml_path=project_root()
            / "configs/datasets/voc.yaml",
            train_images=root / "voc/images/train",
            val_images=root / "voc/images/val",
            test_images=root / "voc/images/test",
            train_labels=root / "voc/labels/train",
            val_labels=root / "voc/labels/val",
            test_labels=root / "voc/labels/test",
            expected_classes=20,
            notes=(
                "Recommended merged VOC2007+2012 training protocol."
            ),
        ),
        "bdd100k": DatasetSpec(
            dataset_id="bdd100k",
            name="BDD100K Detection",
            task="autonomous-driving detection",
            root=root / "bdd100k",
            yaml_path=project_root()
            / "configs/datasets/bdd100k.yaml",
            train_images=root / "bdd100k/images/train",
            val_images=root / "bdd100k/images/val",
            test_images=root / "bdd100k/images/test",
            train_labels=root / "bdd100k/labels/train",
            val_labels=root / "bdd100k/labels/val",
            test_labels=None,
            expected_classes=10,
            notes=(
                "Driving scenes and natural weather/illumination shifts."
            ),
        ),
        "visdrone2019": DatasetSpec(
            dataset_id="visdrone2019",
            name="VisDrone2019-DET",
            task="aerial small-object detection",
            root=root / "visdrone2019",
            yaml_path=project_root()
            / "configs/datasets/visdrone2019.yaml",
            train_images=root / "visdrone2019/images/train",
            val_images=root / "visdrone2019/images/val",
            test_images=root / "visdrone2019/images/test",
            train_labels=root / "visdrone2019/labels/train",
            val_labels=root / "visdrone2019/labels/val",
            test_labels=root / "visdrone2019/labels/test",
            expected_classes=10,
            notes=(
                "Aerial benchmark emphasizing dense small objects."
            ),
        ),
        "exdark": DatasetSpec(
            dataset_id="exdark",
            name="ExDark",
            task="low-light object detection",
            root=root / "exdark",
            yaml_path=project_root()
            / "configs/datasets/exdark.yaml",
            train_images=root / "exdark/images/train",
            val_images=root / "exdark/images/val",
            test_images=root / "exdark/images/test",
            train_labels=root / "exdark/labels/train",
            val_labels=root / "exdark/labels/val",
            test_labels=root / "exdark/labels/test",
            expected_classes=12,
            notes=(
                "Low-light stress-test dataset. Requires explicit "
                "reproducible split because no universal YOLO split "
                "is assumed here."
            ),
        ),
    }


def required_paths(spec: DatasetSpec) -> list[Path]:
    return [
        spec.train_images,
        spec.val_images,
        spec.train_labels,
        spec.val_labels,
        spec.yaml_path,
    ]


def dataset_status(
    spec: DatasetSpec,
    *,
    declared_classes: int | None = None,
    train_image_count: int = 0,
    val_image_count: int = 0,
    train_label_count: int = 0,
    val_label_count: int = 0,
) -> DatasetStatus:
    existing = [
        path.exists()
        for path in required_paths(spec)
    ]

    class_map_valid = (
        spec.expected_classes is None
        or declared_classes == spec.expected_classes
    )

    data_nonempty = all(
        count > 0
        for count in (
            train_image_count,
            val_image_count,
            train_label_count,
            val_label_count,
        )
    )

    if all(existing) and class_map_valid and data_nonempty:
        return "ready"

    if any(existing):
        return "partial"

    return "missing"
