from __future__ import annotations

import copy
import io
import json
import traceback
from pathlib import Path
from typing import Any, Callable

import torch
from ultralytics.utils.torch_utils import ModelEMA

from src.qvisionframe.quantum_detection_model import (
    QuantumGatedDetectionModel,
)


def make_model(
    *,
    device: str = "cpu",
) -> QuantumGatedDetectionModel:
    torch.manual_seed(42)

    model = QuantumGatedDetectionModel(
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

    return model.to(device)


def attempt(
    name: str,
    function: Callable[[], Any],
) -> dict[str, Any]:
    try:
        result = function()

        print(f"{name:<44}: PASSED")
        print("  result:", result)

        return {
            "passed": True,
            "result": result,
            "error_type": None,
            "error": None,
            "traceback": None,
        }

    except Exception as error:
        print(f"{name:<44}: FAILED")
        print(
            "  error :",
            type(error).__name__,
            str(error),
        )

        return {
            "passed": False,
            "result": None,
            "error_type": type(error).__name__,
            "error": str(error),
            "traceback": traceback.format_exc(),
        }


def describe_copy(
    module: torch.nn.Module,
) -> dict[str, Any]:
    copied = copy.deepcopy(module)

    return {
        "type": type(copied).__name__,
        "state_tensors": len(
            copied.state_dict()
        ),
    }


def save_full_object(
    module: torch.nn.Module,
) -> dict[str, Any]:
    memory = io.BytesIO()

    torch.save(
        module,
        memory,
    )

    byte_size = memory.tell()
    memory.seek(0)

    loaded = torch.load(
        memory,
        map_location="cpu",
        weights_only=False,
    )

    return {
        "serialized_bytes": byte_size,
        "loaded_type": type(loaded).__name__,
        "state_tensors": len(
            loaded.state_dict()
        ),
    }


def run_gate_forward(
    model: QuantumGatedDetectionModel,
) -> dict[str, Any]:
    model.eval()

    device = next(
        model.parameters()
    ).device

    model.reset_gate_observations()

    with torch.no_grad():
        output = model(
            torch.randn(
                1,
                3,
                320,
                320,
                dtype=torch.float32,
                device=device,
            )
        )

    return {
        "output_type": type(output).__name__,
        "hook_calls": model.hook_calls,
        "feature_shape": (
            model.last_feature_shape
        ),
        "quantum_device": str(
            model.gate.quantum_weights.device
        ),
    }


def ema_update_check(
    model: QuantumGatedDetectionModel,
) -> dict[str, Any]:
    ema = ModelEMA(model)

    # Raw model forward materializes its quantum simulator.
    run_gate_forward(model)

    ema.update(model)

    return {
        "ema": ema,
        "raw_hook_calls": model.hook_calls,
        "ema_hook_calls": ema.ema.hook_calls,
    }


def main() -> None:
    print(
        "===== QUANTUM POST-FORWARD COPY AUDIT ====="
    )

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is unavailable"
        )

    root = Path(__file__).resolve().parents[1]
    report: dict[str, Any] = {}

    # ---------------------------------------------------------
    # CPU model
    # ---------------------------------------------------------
    print()
    print("===== CPU MODEL =====")

    cpu_model = make_model(device="cpu")

    report["cpu_deepcopy_before_forward"] = attempt(
        "CPU deepcopy before forward",
        lambda: describe_copy(cpu_model),
    )

    report["cpu_save_before_forward"] = attempt(
        "CPU torch.save before forward",
        lambda: save_full_object(cpu_model),
    )

    cpu_forward = run_gate_forward(cpu_model)

    print(
        "CPU forward materialization             :",
        cpu_forward,
    )

    report["cpu_deepcopy_after_forward"] = attempt(
        "CPU deepcopy after forward",
        lambda: describe_copy(cpu_model),
    )

    report["cpu_save_after_forward"] = attempt(
        "CPU torch.save after forward",
        lambda: save_full_object(cpu_model),
    )

    # ---------------------------------------------------------
    # CUDA model
    # ---------------------------------------------------------
    print()
    print("===== CUDA MODEL =====")

    cuda_model = make_model(device="cuda:0")

    report["cuda_deepcopy_before_forward"] = attempt(
        "CUDA deepcopy before forward",
        lambda: describe_copy(cuda_model),
    )

    report["cuda_save_before_forward"] = attempt(
        "CUDA torch.save before forward",
        lambda: save_full_object(cuda_model),
    )

    cuda_forward = run_gate_forward(cuda_model)

    print(
        "CUDA forward materialization            :",
        cuda_forward,
    )

    report["cuda_deepcopy_after_forward"] = attempt(
        "CUDA deepcopy after forward",
        lambda: describe_copy(cuda_model),
    )

    report["cuda_save_after_forward"] = attempt(
        "CUDA torch.save after forward",
        lambda: save_full_object(cuda_model),
    )

    # ---------------------------------------------------------
    # EMA lifecycle
    # ---------------------------------------------------------
    print()
    print("===== EMA LIFECYCLE =====")

    ema_source = make_model(
        device="cuda:0"
    )

    ema_bundle = ema_update_check(
        ema_source
    )

    ema = ema_bundle["ema"]

    print(
        "Raw hook calls                         :",
        ema_bundle["raw_hook_calls"],
    )
    print(
        "EMA hook calls before EMA forward      :",
        ema_bundle["ema_hook_calls"],
    )

    report["ema_copy_after_raw_forward"] = attempt(
        "Deepcopy EMA after raw model forward",
        lambda: describe_copy(ema.ema),
    )

    report["ema_save_after_raw_forward"] = attempt(
        "torch.save EMA after raw model forward",
        lambda: save_full_object(ema.ema),
    )

    ema_forward = run_gate_forward(
        ema.ema
    )

    print(
        "EMA forward materialization             :",
        ema_forward,
    )

    report["ema_copy_after_ema_forward"] = attempt(
        "Deepcopy EMA after EMA forward",
        lambda: describe_copy(ema.ema),
    )

    report["ema_save_after_ema_forward"] = attempt(
        "torch.save EMA after EMA forward",
        lambda: save_full_object(ema.ema),
    )

    output_path = (
        root
        / "outputs/inventory/"
        "quantum_post_forward_copy_audit.json"
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path.write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )

    print()
    print("Report:", output_path)
    print(
        "QUANTUM POST-FORWARD COPY AUDIT: COMPLETED"
    )


if __name__ == "__main__":
    main()
