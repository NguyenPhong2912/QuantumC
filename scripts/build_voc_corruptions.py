from __future__ import annotations

import argparse
import hashlib
import io
import shutil
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

from dataset_common import materialize_file, write_json


CORRUPTIONS = (
    "gaussian_noise",
    "shot_noise",
    "impulse_noise",
    "gaussian_blur",
    "motion_blur",
    "fog",
    "brightness_down",
    "jpeg",
)
SEVERITIES = (1, 2, 3, 4, 5)
EXPECTED_VAL_IMAGES = 4952


def deterministic_seed(
    base_seed: int,
    image_name: str,
    corruption: str,
    severity: int,
) -> int:
    digest = hashlib.sha256(
        f"{base_seed}:{image_name}:{corruption}:{severity}".encode()
    ).digest()
    return int.from_bytes(digest[:8], "big")


def corrupt_image(
    source: Image.Image,
    corruption: str,
    severity: int,
    rng: np.random.Generator,
) -> Image.Image:
    image = source.convert("RGB")
    array = np.asarray(image).astype(np.float32) / 255.0

    if corruption == "gaussian_noise":
        sigma = (0.04, 0.07, 0.10, 0.14, 0.18)[severity - 1]
        output = array + rng.normal(0.0, sigma, array.shape)
    elif corruption == "shot_noise":
        scale = (120, 60, 30, 15, 8)[severity - 1]
        output = rng.poisson(array * scale) / scale
    elif corruption == "impulse_noise":
        amount = (0.01, 0.02, 0.04, 0.07, 0.10)[severity - 1]
        output = array.copy()
        mask = rng.random(array.shape[:2])
        output[mask < amount / 2.0] = 0.0
        output[mask > 1.0 - amount / 2.0] = 1.0
    elif corruption == "gaussian_blur":
        radius = (1, 2, 3, 4, 6)[severity - 1]
        return image.filter(ImageFilter.GaussianBlur(radius=radius))
    elif corruption == "motion_blur":
        length = (5, 9, 13, 17, 21)[severity - 1]
        kernel = np.zeros((length, length), dtype=np.float32)
        kernel[length // 2, :] = 1.0 / length
        angle = float(rng.uniform(0.0, 180.0))
        matrix = cv2.getRotationMatrix2D(
            (length / 2.0 - 0.5, length / 2.0 - 0.5),
            angle,
            1.0,
        )
        kernel = cv2.warpAffine(kernel, matrix, (length, length))
        kernel /= max(float(kernel.sum()), 1e-12)
        output = cv2.filter2D(array, -1, kernel)
    elif corruption == "fog":
        alpha = (0.10, 0.18, 0.28, 0.40, 0.55)[severity - 1]
        noise = rng.random(array.shape[:2]).astype(np.float32)
        noise = cv2.GaussianBlur(noise, (0, 0), sigmaX=20 + severity * 8)
        noise = (noise - noise.min()) / max(
            float(noise.max() - noise.min()),
            1e-12,
        )
        fog = np.repeat(noise[:, :, None], 3, axis=2)
        output = array * (1.0 - alpha) + np.maximum(fog, 0.7) * alpha
    elif corruption == "brightness_down":
        factor = (0.80, 0.65, 0.50, 0.35, 0.22)[severity - 1]
        return ImageEnhance.Brightness(image).enhance(factor)
    elif corruption == "jpeg":
        quality = (60, 40, 25, 15, 8)[severity - 1]
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=quality)
        buffer.seek(0)
        with Image.open(buffer) as compressed:
            return compressed.convert("RGB")
    else:
        raise ValueError(f"Unknown corruption: {corruption}")

    output = np.clip(output * 255.0, 0, 255).astype(np.uint8)
    return Image.fromarray(output, mode="RGB")


def build_variant(
    source_images: list[Path],
    source_label_root: Path,
    output_root: Path,
    corruption: str,
    severity: int,
    seed: int,
    workers: int,
    force: bool,
) -> list[str]:
    variant = f"{corruption}/severity_{severity}"
    image_root = output_root / "images" / variant
    label_root = output_root / "labels" / variant
    complete_path = output_root / "manifests" / f"{corruption}_s{severity}.json"

    if complete_path.is_file() and not force:
        payload = __import__("json").loads(
            complete_path.read_text(encoding="utf-8")
        )
        if int(payload.get("images", -1)) == len(source_images):
            print(f"Skipping completed variant: {variant}", flush=True)
            return [
                f"./images/{variant}/{path.stem}.jpg"
                for path in source_images
            ]

    temporary_image_root = image_root.with_name(f"{image_root.name}.tmp")
    temporary_label_root = label_root.with_name(f"{label_root.name}.tmp")
    for path in (temporary_image_root, temporary_label_root):
        if path.exists():
            shutil.rmtree(path)
        path.mkdir(parents=True)

    def process(source_image: Path) -> None:
        source_label = source_label_root / f"{source_image.stem}.txt"
        destination_image = temporary_image_root / f"{source_image.stem}.jpg"
        destination_label = temporary_label_root / f"{source_image.stem}.txt"
        rng = np.random.default_rng(
            deterministic_seed(seed, source_image.name, corruption, severity)
        )
        with Image.open(source_image) as opened:
            corrupted = corrupt_image(opened, corruption, severity, rng)
        corrupted.save(destination_image, format="JPEG", quality=92)
        if destination_image.stat().st_size <= 0:
            raise OSError(f"Empty corruption output: {destination_image}")
        materialize_file(source_label, destination_label)

    with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
        for index, _ in enumerate(executor.map(process, source_images), start=1):
            if index % 1000 == 0:
                print(f"{variant}: generated {index:,} images", flush=True)

    for final_path, temporary_path in (
        (image_root, temporary_image_root),
        (label_root, temporary_label_root),
    ):
        if final_path.exists():
            shutil.rmtree(final_path)
        temporary_path.rename(final_path)

    write_json(
        complete_path,
        {
            "corruption": corruption,
            "severity": severity,
            "seed": seed,
            "images": len(source_images),
        },
    )
    return [f"./images/{variant}/{path.stem}.jpg" for path in source_images]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--corruptions",
        nargs="+",
        choices=CORRUPTIONS,
        default=list(CORRUPTIONS),
    )
    parser.add_argument(
        "--severities",
        nargs="+",
        type=int,
        choices=SEVERITIES,
        default=list(SEVERITIES),
    )
    args = parser.parse_args()

    project_root = args.project_root.resolve()
    source_image_root = project_root / "data/voc/images/val"
    source_label_root = project_root / "data/voc/labels/val"
    output_root = project_root / "data/voc_corruptions"
    source_images = sorted(source_image_root.glob("*"))
    source_images = [
        path
        for path in source_images
        if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png"}
    ]

    if len(source_images) != EXPECTED_VAL_IMAGES:
        raise RuntimeError(
            f"Expected {EXPECTED_VAL_IMAGES} VOC val images, "
            f"found {len(source_images)}"
        )

    (output_root / "manifests").mkdir(parents=True, exist_ok=True)
    manifest_lines: list[str] = []

    for corruption in args.corruptions:
        for severity in args.severities:
            manifest_lines.extend(
                build_variant(
                    source_images,
                    source_label_root,
                    output_root,
                    corruption,
                    severity,
                    args.seed,
                    args.workers,
                    args.force,
                )
            )

    manifest_lines = sorted(set(manifest_lines))
    (output_root / "val.txt").write_text(
        "\n".join(manifest_lines) + "\n",
        encoding="utf-8",
    )
    digest = hashlib.sha256(
        ("\n".join(manifest_lines) + "\n").encode()
    ).hexdigest()
    write_json(
        output_root / "manifests/benchmark.json",
        {
            "source": "VOC validation split",
            "source_images": len(source_images),
            "corruptions": args.corruptions,
            "severities": args.severities,
            "seed": args.seed,
            "generated_images": len(manifest_lines),
            "manifest_sha256": digest,
        },
    )
    print(
        f"VOC CORRUPTION BENCHMARK PASSED: {len(manifest_lines):,} images"
    )


if __name__ == "__main__":
    main()
