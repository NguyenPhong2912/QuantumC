from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

from PIL import Image


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
    )
    parser.add_argument("--verify-every", type=int, default=1)
    args = parser.parse_args()

    project_root = args.project_root.resolve()
    root = project_root / "data/voc_corruptions"
    benchmark = json.loads(
        (root / "manifests/benchmark.json").read_text(encoding="utf-8")
    )
    manifest_text = (root / "val.txt").read_text(encoding="utf-8")
    entries = [line.strip() for line in manifest_text.splitlines() if line.strip()]
    expected = (
        int(benchmark["source_images"])
        * len(benchmark["corruptions"])
        * len(benchmark["severities"])
    )

    assert len(entries) == expected
    assert hashlib.sha256(manifest_text.encode()).hexdigest() == benchmark[
        "manifest_sha256"
    ]

    missing_images = 0
    missing_labels = 0
    unreadable_images = 0

    for index, relative in enumerate(entries):
        image_path = root / relative.removeprefix("./")
        label_path = Path(
            str(image_path).replace(
                f"{os.sep}images{os.sep}",
                f"{os.sep}labels{os.sep}",
                1,
            )
        ).with_suffix(".txt")

        if not image_path.is_file() or image_path.stat().st_size <= 0:
            missing_images += 1
            continue
        if not label_path.is_file():
            missing_labels += 1
        if index % max(1, args.verify_every) == 0:
            try:
                with Image.open(image_path) as image:
                    image.verify()
            except Exception:
                unreadable_images += 1

    result = {
        "images": len(entries),
        "missing_images": missing_images,
        "missing_labels": missing_labels,
        "unreadable_images": unreadable_images,
        "manifest_sha256": benchmark["manifest_sha256"],
    }
    output = project_root / "outputs/qa/voc_corruptions"
    output.mkdir(parents=True, exist_ok=True)
    (output / "dataset_report.json").write_text(
        json.dumps(result, indent=2) + "\n",
        encoding="utf-8",
    )
    assert missing_images == 0
    assert missing_labels == 0
    assert unreadable_images == 0
    print(f"VOC CORRUPTION QA PASSED: {len(entries):,} images")


if __name__ == "__main__":
    main()
