from __future__ import annotations

from pathlib import Path

from ultralytics.data.utils import check_det_dataset


def main() -> None:
    print("===== ULTRALYTICS VOC DATASET CHECK =====")

    project_root = Path(__file__).resolve().parents[1]
    yaml_path = project_root / "configs/datasets/voc.yaml"

    dataset = check_det_dataset(
        str(yaml_path),
        autodownload=False,
    )

    train_path = Path(dataset["train"])
    val_path = Path(dataset["val"])
    names = dataset["names"]
    nc = dataset["nc"]

    print("YAML           :", yaml_path)
    print("Dataset root   :", dataset.get("path"))
    print("Train path     :", train_path)
    print("Validation path:", val_path)
    print("Classes        :", nc)
    print("Class 0        :", names[0])
    print("Class 19       :", names[19])

    assert train_path.is_dir()
    assert val_path.is_dir()
    assert nc == 20
    assert len(names) == 20
    assert names[0] == "aeroplane"
    assert names[19] == "tvmonitor"

    print("ULTRALYTICS DATASET CONFIG: PASSED")


if __name__ == "__main__":
    main()
