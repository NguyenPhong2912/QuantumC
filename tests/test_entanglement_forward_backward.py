from __future__ import annotations

import torch

from src.qvisionframe.config import load_config
from src.qvisionframe.factory import build_hybrid_model


CONFIG_PATHS = {
    "none": "configs/experiments/pqfg_p5_q4_l2_none.yaml",
    "linear": "configs/experiments/pqfg_p5_q4_l2_linear.yaml",
    "ring": "configs/experiments/pqfg_p5_q4_l2_ring.yaml",
}


def grad_norm(parameter: torch.Tensor) -> float:
    if parameter.grad is None:
        return 0.0
    return float(parameter.grad.detach().float().norm().cpu())


def collect_tensors(obj) -> list[torch.Tensor]:
    tensors: list[torch.Tensor] = []

    if torch.is_tensor(obj):
        tensors.append(obj)
    elif isinstance(obj, dict):
        for value in obj.values():
            tensors.extend(collect_tensors(value))
    elif isinstance(obj, (list, tuple)):
        for value in obj:
            tensors.extend(collect_tensors(value))

    return tensors


def main() -> None:
    print("===== ENTANGLEMENT FORWARD/BACKWARD CHECK =====")

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")

    device = torch.device("cuda:0")
    results: dict[str, dict[str, object]] = {}

    for topology, path in CONFIG_PATHS.items():
        # Reset seed so all variants start from identical classical
        # and quantum parameters. Only entanglement differs.
        torch.manual_seed(42)

        config = load_config(path)
        hybrid = build_hybrid_model(
            config=config,
            training=True,
        )

        hybrid.zero_grad(set_to_none=True)

        x = torch.randn(
            1,
            3,
            320,
            320,
            dtype=torch.float32,
            device=device,
            requires_grad=True,
        )

        output = hybrid(x)

        differentiable = [
            tensor
            for tensor in collect_tensors(output)
            if tensor.is_floating_point() and tensor.requires_grad
        ]

        if not differentiable:
            raise RuntimeError(
                f"No differentiable tensors for topology {topology}"
            )

        loss = sum(
            tensor.float().square().mean()
            for tensor in differentiable
        )

        loss.backward()

        quantum_grad = grad_norm(
            hybrid.pqfg.quantum_weights
        )
        projector_grad = grad_norm(
            hybrid.pqfg.projector[0].weight
        )
        decoder_grad = grad_norm(
            hybrid.pqfg.decoder.weight
        )
        alpha_grad = grad_norm(
            hybrid.pqfg.alpha
        )

        # Direct circuit output for a fixed descriptor-like angle vector.
        angles = torch.tensor(
            [[0.15, -0.30, 0.45, -0.60]],
            dtype=torch.float64,
            device="cpu",
        )

        with torch.no_grad():
            measurement = (
                hybrid.pqfg.quantum_forward(angles)
                .detach()
                .cpu()
            )

        results[topology] = {
            "loss": float(loss.detach()),
            "measurement": measurement,
            "quantum_grad": quantum_grad,
            "projector_grad": projector_grad,
            "decoder_grad": decoder_grad,
            "alpha_grad": alpha_grad,
            "feature_shape": hybrid.last_feature_shape,
        }

        print("Topology             :", topology)
        print("Feature shape        :", hybrid.last_feature_shape)
        print("Loss                 :", float(loss.detach()))
        print(
            "Measurement          :",
            measurement.flatten().tolist(),
        )
        print("Projector grad norm  :", projector_grad)
        print("Quantum grad norm    :", quantum_grad)
        print("Decoder grad norm    :", decoder_grad)
        print("Alpha grad norm      :", alpha_grad)
        print("-" * 72)

        assert hybrid.last_feature_shape == (1, 256, 10, 10)
        assert torch.isfinite(loss)
        assert torch.isfinite(measurement).all()
        assert projector_grad > 0.0
        assert quantum_grad > 0.0
        assert decoder_grad > 0.0
        assert alpha_grad > 0.0

        hybrid.close()

    none_measurement = results["none"]["measurement"]
    linear_measurement = results["linear"]["measurement"]
    ring_measurement = results["ring"]["measurement"]

    none_linear_equal = torch.allclose(
        none_measurement,
        linear_measurement,
        atol=1.0e-10,
        rtol=1.0e-8,
    )

    linear_ring_equal = torch.allclose(
        linear_measurement,
        ring_measurement,
        atol=1.0e-10,
        rtol=1.0e-8,
    )

    none_ring_equal = torch.allclose(
        none_measurement,
        ring_measurement,
        atol=1.0e-10,
        rtol=1.0e-8,
    )

    print("None vs linear equal :", none_linear_equal)
    print("Linear vs ring equal :", linear_ring_equal)
    print("None vs ring equal   :", none_ring_equal)

    assert not none_linear_equal
    assert not linear_ring_equal
    assert not none_ring_equal

    print("ENTANGLEMENT F/B     : PASSED")


if __name__ == "__main__":
    main()
