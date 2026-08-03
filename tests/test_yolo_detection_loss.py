import torch
from ultralytics import YOLO
from ultralytics.cfg import get_cfg
from ultralytics.utils import DEFAULT_CFG


def main() -> None:
    print("===== YOLO DETECTION LOSS CHECK =====")

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")

    torch.manual_seed(42)

    device = torch.device("cuda:0")

    model = YOLO("yolo11n.yaml")
    network = model.model.to(device).train()

    # Direct construction from model YAML leaves network.args as a dict.
    # The normal Trainer path converts default training configuration into
    # a namespace containing box, cls, dfl and other loss hyperparameters.
    network.args = get_cfg(
        cfg=DEFAULT_CFG,
        overrides=model.overrides,
    )

    # Force lazy criterion creation after assigning normalized arguments.
    network.criterion = None
    network.zero_grad(set_to_none=True)

    batch_size = 2
    image_size = 320

    # Ultralytics expects float images in [0, 1].
    images = torch.rand(
        batch_size,
        3,
        image_size,
        image_size,
        dtype=torch.float32,
        device=device,
    )

    # Four objects total: two objects per image.
    # Boxes use normalized xywh format.
    cls = torch.tensor(
        [[0.0], [5.0], [2.0], [7.0]],
        dtype=torch.float32,
        device=device,
    )

    bboxes = torch.tensor(
        [
            [0.30, 0.35, 0.20, 0.25],
            [0.70, 0.65, 0.15, 0.20],
            [0.40, 0.45, 0.25, 0.30],
            [0.75, 0.30, 0.18, 0.22],
        ],
        dtype=torch.float32,
        device=device,
    )

    batch_idx = torch.tensor(
        [0, 0, 1, 1],
        dtype=torch.long,
        device=device,
    )

    batch = {
        "img": images,
        "cls": cls,
        "bboxes": bboxes,
        "batch_idx": batch_idx,
    }

    result = network(batch)

    if not isinstance(result, tuple) or len(result) != 2:
        raise RuntimeError(
            f"Expected (loss, loss_items), received {type(result).__name__}"
        )

    loss_components, loss_items = result

    # Ultralytics 8.4.x returns the three differentiable detection-loss
    # components: box, classification and DFL. Sum them to obtain the
    # scalar objective used for this backward smoke test.
    if loss_components.ndim == 0:
        total_loss = loss_components
    else:
        total_loss = loss_components.sum()

    print("GPU                 :", torch.cuda.get_device_name(0))
    print("Image shape         :", tuple(images.shape))
    print("Object count        :", int(cls.shape[0]))
    print("Loss shape          :", tuple(loss_components.shape))
    print(
        "Loss components     :",
        loss_components.detach().cpu().tolist(),
    )
    print("Total loss          :", float(total_loss.detach()))
    print("Loss items shape    :", tuple(loss_items.shape))
    print("Loss items          :", loss_items.detach().cpu().tolist())
    print(
        "Loss finite         :",
        bool(torch.isfinite(loss_components).all()),
    )
    print(
        "Loss items finite   :",
        bool(torch.isfinite(loss_items).all()),
    )

    total_loss.backward()

    parameters_with_grad = [
        parameter
        for parameter in network.parameters()
        if parameter.requires_grad and parameter.grad is not None
    ]

    finite_grads = all(
        bool(torch.isfinite(parameter.grad).all())
        for parameter in parameters_with_grad
    )

    total_grad_norm = torch.sqrt(
        sum(
            parameter.grad.detach().float().square().sum()
            for parameter in parameters_with_grad
        )
    )

    print("Params with grad    :", len(parameters_with_grad))
    print("Param grads finite  :", finite_grads)
    print("Total grad norm     :", float(total_grad_norm))

    assert torch.isfinite(loss_components).all()
    assert torch.isfinite(loss_items).all()
    assert float(total_loss.detach()) > 0.0
    assert parameters_with_grad
    assert finite_grads
    assert float(total_grad_norm) > 0.0

    print("YOLO DETECTION LOSS : PASSED")


if __name__ == "__main__":
    main()
