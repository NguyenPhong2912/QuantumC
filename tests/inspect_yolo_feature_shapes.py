from __future__ import annotations

from typing import Any

import torch
from ultralytics import YOLO


def extract_shapes(output: Any) -> list[tuple[int, ...]]:
    """Recursively obtain tensor shapes from a nested layer output."""
    shapes: list[tuple[int, ...]] = []

    if torch.is_tensor(output):
        shapes.append(tuple(output.shape))
    elif isinstance(output, dict):
        for value in output.values():
            shapes.extend(extract_shapes(value))
    elif isinstance(output, (list, tuple)):
        for value in output:
            shapes.extend(extract_shapes(value))

    return shapes


def main() -> None:
    print("===== YOLO FEATURE SHAPE INSPECTION =====")

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")

    torch.manual_seed(42)

    device = torch.device("cuda:0")
    wrapper = YOLO("yolo11n.yaml")
    network = wrapper.model.to(device).eval()

    records: dict[int, dict[str, object]] = {}
    hooks = []

    for index, layer in enumerate(network.model):
        def make_hook(layer_index: int):
            def hook(module, inputs, output):
                records[layer_index] = {
                    "type": module.__class__.__name__,
                    "from": getattr(module, "f", None),
                    "params": getattr(module, "np", None),
                    "shapes": extract_shapes(output),
                }

            return hook

        hooks.append(
            layer.register_forward_hook(make_hook(index))
        )

    x = torch.randn(
        1,
        3,
        640,
        640,
        dtype=torch.float32,
        device=device,
    )

    with torch.inference_mode():
        output = network(x)

    for handle in hooks:
        handle.remove()

    print("Input shape:", tuple(x.shape))
    print()
    print(
        f"{'IDX':>3}  {'FROM':>10}  {'MODULE':<24} "
        f"{'PARAMS':>12}  OUTPUT SHAPES"
    )
    print("-" * 100)

    for index in sorted(records):
        record = records[index]

        print(
            f"{index:>3}  "
            f"{str(record['from']):>10}  "
            f"{str(record['type']):<24} "
            f"{str(record['params']):>12}  "
            f"{record['shapes']}"
        )

    feature_candidates = []

    for index, record in records.items():
        for shape in record["shapes"]:
            if len(shape) != 4:
                continue

            _, channels, height, width = shape

            if height == width and height in {80, 40, 20}:
                level = {
                    80: "P3 / stride 8",
                    40: "P4 / stride 16",
                    20: "P5 / stride 32",
                }[height]

                feature_candidates.append(
                    (
                        index,
                        record["type"],
                        level,
                        channels,
                        height,
                        width,
                    )
                )

    print()
    print("===== P3/P4/P5 CANDIDATES =====")

    for (
        index,
        module_type,
        level,
        channels,
        height,
        width,
    ) in feature_candidates:
        print(
            f"Layer {index:>2}: "
            f"{module_type:<20} "
            f"{level:<16} "
            f"shape=(1, {channels}, {height}, {width})"
        )

    print()
    print("Final output type:", type(output).__name__)

    assert records
    assert feature_candidates

    print("YOLO FEATURE INSPECTION: PASSED")


if __name__ == "__main__":
    main()
