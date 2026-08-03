from __future__ import annotations

import gc

import torch
from ultralytics.utils.torch_utils import ModelEMA

from src.qvisionframe.quantum_detection_model import (
    QuantumGatedDetectionModel,
)


def make_model() -> QuantumGatedDetectionModel:
    torch.manual_seed(42)

    return QuantumGatedDetectionModel(
        cfg="yolo11n.yaml",
        nc=20,
        baseline_id="B8",
        target_layer=10,
        channels=256,
        n_qubits=4,
        n_layers=2,
        hidden_dim=64,
        alpha_init=1.0e-3,
        gamma=0.5,
        verbose=False,
    )


def classical_parameter_dtypes(
    model: QuantumGatedDetectionModel,
) -> set[torch.dtype]:
    return {
        parameter.dtype
        for name, parameter in model.named_parameters()
        if name != "gate.quantum_weights"
        and parameter.is_floating_point()
    }


def run_forward(
    model: QuantumGatedDetectionModel,
) -> torch.Tensor:
    device = next(model.parameters()).device

    model.reset_gate_observations()
    model.eval()

    with torch.no_grad():
        output = model(
            torch.randn(
                1,
                3,
                320,
                320,
                dtype=next(
                    parameter.dtype
                    for name, parameter
                    in model.named_parameters()
                    if (
                        name
                        != "gate.quantum_weights"
                        and parameter.ndim > 1
                    )
                ),
                device=device,
            )
        )

    if isinstance(output, tuple):
        predictions = output[0]
    else:
        predictions = output

    return predictions


def main() -> None:
    print(
        "===== QUANTUM PRECISION CONTRACT ====="
    )

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is unavailable"
        )

    # ---------------------------------------------------------
    # Raw quantum model
    # ---------------------------------------------------------
    model = make_model().cuda()

    print()
    print("Initial quantum dtype :", model.quantum_weights.dtype)
    print("Initial quantum device:", model.quantum_weights.device)

    assert model.quantum_weights.dtype == torch.float64
    assert model.quantum_weights.device.type == "cuda"

    # Same conversion used by BaseValidator training branch.
    model.float()

    print(
        "After model.float() quantum dtype:",
        model.quantum_weights.dtype,
    )
    print(
        "Classical dtypes after float       :",
        classical_parameter_dtypes(model),
    )

    assert model.quantum_weights.dtype == torch.float64
    assert classical_parameter_dtypes(model) == {
        torch.float32
    }

    output_fp32 = run_forward(model)

    print(
        "FP32 forward output dtype/device   :",
        output_fp32.dtype,
        output_fp32.device,
    )
    print(
        "FP32 forward hook calls            :",
        model.hook_calls,
    )

    assert output_fp32.dtype == torch.float32
    assert output_fp32.device.type == "cuda"
    assert model.hook_calls == 1
    assert model.quantum_weights.dtype == torch.float64

    # ---------------------------------------------------------
    # Explicit half conversion safety
    # ---------------------------------------------------------
    half_model = make_model().cuda()
    half_model.half()

    print()
    print(
        "After model.half() quantum dtype   :",
        half_model.quantum_weights.dtype,
    )
    print(
        "Classical dtypes after half        :",
        classical_parameter_dtypes(half_model),
    )

    assert half_model.quantum_weights.dtype == torch.float64
    assert classical_parameter_dtypes(half_model) == {
        torch.float16
    }

    output_fp16 = run_forward(half_model)

    print(
        "FP16 forward output dtype/device   :",
        output_fp16.dtype,
        output_fp16.device,
    )
    print(
        "FP16 forward hook calls            :",
        half_model.hook_calls,
    )

    assert output_fp16.dtype == torch.float16
    assert output_fp16.device.type == "cuda"
    assert half_model.hook_calls == 1
    assert half_model.quantum_weights.dtype == torch.float64

    # ---------------------------------------------------------
    # EMA validation-like lifecycle
    # ---------------------------------------------------------
    ema_source = make_model().cuda()
    ema = ModelEMA(ema_source)

    print()
    print(
        "EMA quantum dtype before float     :",
        ema.ema.quantum_weights.dtype,
    )

    assert ema.ema.quantum_weights.dtype == torch.float64

    # BaseValidator performs this in training mode.
    ema.ema.float()

    print(
        "EMA quantum dtype after float      :",
        ema.ema.quantum_weights.dtype,
    )
    print(
        "EMA classical dtypes after float   :",
        classical_parameter_dtypes(ema.ema),
    )

    assert ema.ema.quantum_weights.dtype == torch.float64
    assert classical_parameter_dtypes(ema.ema) == {
        torch.float32
    }

    ema_output = run_forward(ema.ema)

    print(
        "EMA output dtype/device            :",
        ema_output.dtype,
        ema_output.device,
    )
    print(
        "EMA hook calls                     :",
        ema.ema.hook_calls,
    )
    print(
        "EMA quantum dtype after forward    :",
        ema.ema.quantum_weights.dtype,
    )

    assert ema_output.dtype == torch.float32
    assert ema_output.device.type == "cuda"
    assert ema.ema.hook_calls == 1
    assert ema.ema.quantum_weights.dtype == torch.float64

    # EMA remains numerically compatible with another update.
    ema.update(ema_source)

    print(
        "EMA quantum dtype after update     :",
        ema.ema.quantum_weights.dtype,
    )
    print(
        "EMA updates                        :",
        ema.updates,
    )

    assert ema.ema.quantum_weights.dtype == torch.float64
    assert ema.updates == 1

    for item in (
        model,
        half_model,
        ema_source,
        ema.ema,
    ):
        item.close_gate_hook()

    del model
    del half_model
    del ema_source
    del ema
    del output_fp32
    del output_fp16
    del ema_output

    gc.collect()
    torch.cuda.empty_cache()

    print()
    print(
        "QUANTUM PRECISION CONTRACT: PASSED"
    )


if __name__ == "__main__":
    main()
