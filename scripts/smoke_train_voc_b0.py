from __future__ import annotations

import json
import os
import random
from pathlib import Path

import numpy as np
import torch
from ultralytics import YOLO


SEED = 42


def set_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main() -> None:
    print("===== VOC B0 SMOKE TRAINING =====")

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")

    set_seed(SEED)

    project_root = Path(__file__).resolve().parents[1]
    data_yaml = (
        project_root / "configs/datasets/voc.yaml"
    ).resolve()

    output_project = (
        project_root
        / "outputs"
        / os.environ.get("QVF_SERVER", "FITLAB-01")
        / "smoke_training"
    )

    run_name = "voc_b0_yolo11n_seed42"

    print("Host              :", os.uname().nodename)
    print("GPU               :", torch.cuda.get_device_name(0))
    print("Ultralytics YAML  :", data_yaml)
    print("Output project    :", output_project)
    print("Run name          :", run_name)
    print("Training fraction :", 0.01)
    print("Epochs            :", 1)
    print("Image size        :", 320)
    print("Batch size        :", 4)

    model = YOLO("yolo11n.yaml")

    results = model.train(
        data=str(data_yaml),
        epochs=1,
        imgsz=320,
        batch=4,
        device=0,
        workers=2,
        fraction=0.01,
        cache=False,
        pretrained=False,
        optimizer="SGD",
        lr0=0.01,
        seed=SEED,
        deterministic=True,
        amp=True,
        val=True,
        plots=False,
        save=True,
        save_period=1,
        project=str(output_project),
        name=run_name,
        exist_ok=True,
        verbose=True,
    )

    save_dir = Path(results.save_dir)
    weights_dir = save_dir / "weights"
    last_checkpoint = weights_dir / "last.pt"
    best_checkpoint = weights_dir / "best.pt"
    results_csv = save_dir / "results.csv"
    args_yaml = save_dir / "args.yaml"

    summary = {
        "save_dir": str(save_dir),
        "last_checkpoint_exists": last_checkpoint.is_file(),
        "best_checkpoint_exists": best_checkpoint.is_file(),
        "results_csv_exists": results_csv.is_file(),
        "args_yaml_exists": args_yaml.is_file(),
        "gpu": torch.cuda.get_device_name(0),
        "seed": SEED,
        "fraction": 0.01,
        "epochs": 1,
        "imgsz": 320,
        "batch": 4,
    }

    summary_path = save_dir / "smoke_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )

    print()
    print("===== SMOKE TRAINING ARTIFACTS =====")
    print("Save directory :", save_dir)
    print("last.pt        :", last_checkpoint.is_file())
    print("best.pt        :", best_checkpoint.is_file())
    print("results.csv    :", results_csv.is_file())
    print("args.yaml      :", args_yaml.is_file())
    print("Summary        :", summary_path)

    assert last_checkpoint.is_file()
    assert best_checkpoint.is_file()
    assert results_csv.is_file()
    assert args_yaml.is_file()

    print("VOC B0 SMOKE TRAINING: PASSED")


if __name__ == "__main__":
    main()
