from __future__ import annotations

import json
import os
from pathlib import Path

import torch

from src.qvisionframe.quantum_checkpoint import (
    load_quantum_checkpoint,
    reconstruct_quantum_model,
)


ROOT = Path(__file__).resolve().parents[1]

SOURCE_CHECKPOINT = (
    ROOT
    / "outputs/FITLAB-01/smoke_training/"
    "voc_b8_quantum_smoke_seed42/"
    "weights/last.pt"
)

REPORT_PATH = (
    ROOT
    / "outputs/inventory/"
    "cross_server_b8_checkpoint_audit.json"
)


def optimizer_tensor_inventory(
    value,
) -> tuple[int, set[str], set[str]]:
    tensor_count = 0
    dtypes: set[str] = set()
    devices: set[str] = set()

    if isinstance(value, torch.Tensor):
        return (
            1,
            {str(value.dtype)},
            {str(value.device)},
        )

    if isinstance(value, dict):
        for item in value.values():
            count, item_dtypes, item_devices = (
                optimizer_tensor_inventory(item)
            )

            tensor_count += count
            dtypes.update(item_dtypes)
            devices.update(item_devices)

    elif isinstance(value, (list, tuple)):
        for item in value:
            count, item_dtypes, item_devices = (
                optimizer_tensor_inventory(item)
            )

            tensor_count += count
            dtypes.update(item_dtypes)
            devices.update(item_devices)

    return tensor_count, dtypes, devices


def main() -> None:
    print(
        "===== CROSS-SERVER B8 CHECKPOINT AUDIT ====="
    )

    server = os.environ.get(
        "QVF_SERVER",
        "UNKNOWN",
    )

    print("Current server       :", server)
    print("Source checkpoint    :", SOURCE_CHECKPOINT)
    print("Checkpoint exists    :", SOURCE_CHECKPOINT.exists())

    assert server == "FITLAB-02"
    assert SOURCE_CHECKPOINT.exists()
    assert torch.cuda.is_available()

    payload = load_quantum_checkpoint(
        SOURCE_CHECKPOINT,
        map_location="cpu",
    )

    print("Checkpoint format    :", payload["qvf_format"])
    print("Checkpoint epoch     :", payload["epoch"])
    print("Checkpoint baseline  :", payload["model_metadata"]["gate"]["baseline_id"])
    print("Checkpoint entangle  :", payload["model_metadata"]["gate"]["entanglement"])
    print("EMA updates          :", payload["ema_updates"])

    assert payload["qvf_format"] == "qvisionframe.quantum.v1"
    assert payload["epoch"] == 0
    assert payload["model_metadata"]["gate"]["baseline_id"] == "B8"
    assert payload["model_metadata"]["gate"]["entanglement"] == "ring"
    assert payload["ema_updates"] == 16

    raw_model = reconstruct_quantum_model(
        payload,
        state_key="model_state_dict",
        device="cuda:0",
        strict=True,
        verbose=False,
    ).eval()

    ema_model = reconstruct_quantum_model(
        payload,
        state_key="ema_state_dict",
        device="cuda:0",
        strict=True,
        verbose=False,
    ).eval()

    print("Raw model device     :", next(raw_model.parameters()).device)
    print("EMA model device     :", next(ema_model.parameters()).device)
    print("Raw quantum dtype    :", raw_model.quantum_weights.dtype)
    print("EMA quantum dtype    :", ema_model.quantum_weights.dtype)
    print("Raw quantum device   :", raw_model.quantum_weights.device)
    print("EMA quantum device   :", ema_model.quantum_weights.device)

    assert raw_model.quantum_weights.dtype == torch.float64
    assert ema_model.quantum_weights.dtype == torch.float64
    assert raw_model.quantum_weights.device.type == "cuda"
    assert ema_model.quantum_weights.device.type == "cuda"

    raw_model.reset_gate_observations()
    ema_model.reset_gate_observations()

    image = torch.randn(
        1,
        3,
        320,
        320,
        dtype=torch.float32,
        device="cuda:0",
    )

    with torch.no_grad():
        raw_output = raw_model(image)
        ema_output = ema_model(image)

    raw_predictions = (
        raw_output[0]
        if isinstance(raw_output, tuple)
        else raw_output
    )

    ema_predictions = (
        ema_output[0]
        if isinstance(ema_output, tuple)
        else ema_output
    )

    print("Raw hook calls       :", raw_model.hook_calls)
    print("EMA hook calls       :", ema_model.hook_calls)
    print("Raw feature shape    :", raw_model.last_feature_shape)
    print("EMA feature shape    :", ema_model.last_feature_shape)
    print("Raw output finite    :", bool(torch.isfinite(raw_predictions).all()))
    print("EMA output finite    :", bool(torch.isfinite(ema_predictions).all()))

    assert raw_model.hook_calls == 1
    assert ema_model.hook_calls == 1
    assert raw_model.last_feature_shape == (1, 256, 10, 10)
    assert ema_model.last_feature_shape == (1, 256, 10, 10)
    assert bool(torch.isfinite(raw_predictions).all())
    assert bool(torch.isfinite(ema_predictions).all())

    optimizer_state = payload[
        "optimizer_state_dict"
    ]

    assert optimizer_state is not None
    assert len(optimizer_state["state"]) == 263

    tensor_count, dtypes, devices = (
        optimizer_tensor_inventory(
            optimizer_state
        )
    )

    print("Optimizer entries    :", len(optimizer_state["state"]))
    print("Optimizer tensors    :", tensor_count)
    print("Optimizer dtypes     :", sorted(dtypes))
    print("Optimizer devices    :", sorted(devices))

    # load_quantum_checkpoint(map_location="cpu") must place all
    # serialized optimizer tensors on CPU before optimizer restoration.
    assert devices == {"cpu"}
    assert tensor_count > 0

    report = {
        "current_server": server,
        "source_server": "FITLAB-01",
        "source_checkpoint": str(
            SOURCE_CHECKPOINT
        ),
        "format": payload["qvf_format"],
        "epoch": payload["epoch"],
        "baseline": payload[
            "model_metadata"
        ]["gate"]["baseline_id"],
        "entanglement": payload[
            "model_metadata"
        ]["gate"]["entanglement"],
        "ema_updates": payload["ema_updates"],
        "raw_quantum_dtype": str(
            raw_model.quantum_weights.dtype
        ),
        "ema_quantum_dtype": str(
            ema_model.quantum_weights.dtype
        ),
        "raw_quantum_device": str(
            raw_model.quantum_weights.device
        ),
        "ema_quantum_device": str(
            ema_model.quantum_weights.device
        ),
        "raw_hook_calls": raw_model.hook_calls,
        "ema_hook_calls": ema_model.hook_calls,
        "optimizer_entries": len(
            optimizer_state["state"]
        ),
        "optimizer_tensor_count": tensor_count,
        "optimizer_dtypes": sorted(dtypes),
        "optimizer_devices": sorted(devices),
    }

    REPORT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    REPORT_PATH.write_text(
        json.dumps(report, indent=2)
        + "\n",
        encoding="utf-8",
    )

    raw_model.close_gate_hook()
    ema_model.close_gate_hook()

    print("Report               :", REPORT_PATH)
    print()
    print(
        "CROSS-SERVER B8 CHECKPOINT AUDIT: PASSED"
    )


if __name__ == "__main__":
    main()
