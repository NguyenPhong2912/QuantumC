import time

import torch
from ultralytics import YOLO


def main() -> None:
    print("===== YOLO ENVIRONMENT CHECK =====")
    print("Torch version       :", torch.__version__)
    print("CUDA available      :", torch.cuda.is_available())

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")

    print("GPU                 :", torch.cuda.get_device_name(0))

    # Khởi tạo kiến trúc, không tải pretrained weights.
    model = YOLO("yolo11n.yaml")
    network = model.model.cuda().eval()

    total_params = sum(p.numel() for p in network.parameters())
    trainable_params = sum(
        p.numel() for p in network.parameters() if p.requires_grad
    )

    x = torch.randn(1, 3, 320, 320, device="cuda")

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()

    start = time.perf_counter()

    with torch.inference_mode():
        output = network(x)

    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    peak_memory_mib = torch.cuda.max_memory_allocated() / 1024**2

    print("Model class         :", type(network).__name__)
    print("Total parameters    :", total_params)
    print("Trainable parameters:", trainable_params)
    print("Input shape         :", tuple(x.shape))
    print("Output type         :", type(output).__name__)
    print("Elapsed seconds     :", round(elapsed, 6))
    print("Peak memory MiB     :", round(peak_memory_mib, 2))
    print("Output exists       :", output is not None)

    assert total_params > 0
    assert trainable_params > 0
    assert output is not None

    print("YOLO FORWARD TEST   : PASSED")


if __name__ == "__main__":
    main()
