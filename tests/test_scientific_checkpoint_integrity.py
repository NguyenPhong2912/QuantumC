from __future__ import annotations

import copy
from pathlib import Path

import torch

from src.qvisionframe.quantum_checkpoint import (
    load_quantum_checkpoint,
    save_quantum_checkpoint,
)
from src.qvisionframe.scientific_state_hash import (
    attach_scientific_state_manifest,
    verify_scientific_state_manifest,
)


ROOT = Path(__file__).resolve().parents[1]

SOURCE = (
    ROOT
    / "outputs/FITLAB-01/resume_training/"
    "voc_b8_quantum_resume_seed42/"
    "weights/last.pt"
)

OUTPUT_DIRECTORY = (
    ROOT
    / "outputs/inventory/"
    "scientific_integrity"
)

SIGNED_PATH = (
    OUTPUT_DIRECTORY
    / "b8_signed_checkpoint.pt"
)

TAMPERED_PATH = (
    OUTPUT_DIRECTORY
    / "b8_tampered_checkpoint.pt"
)


def main() -> None:
    print(
        "===== SCIENTIFIC CHECKPOINT "
        "INTEGRITY TEST ====="
    )

    assert SOURCE.exists()

    # Existing B8 checkpoint is a legacy file without an embedded
    # manifest. Loading it must remain backward compatible.
    legacy = load_quantum_checkpoint(
        SOURCE,
        map_location="cpu",
    )

    print(
        "Legacy manifest present:",
        "scientific_state" in legacy,
    )

    signed = copy.deepcopy(legacy)

    attach_scientific_state_manifest(
        signed
    )

    manifest = (
        verify_scientific_state_manifest(
            signed,
            required=True,
        )
    )

    assert manifest is not None

    print(
        "Embedded schema        :",
        manifest["schema"],
    )
    print(
        "Embedded algorithm     :",
        manifest["algorithm"],
    )
    print(
        "Embedded digest        :",
        manifest["digest"],
    )
    print(
        "Raw state tensors      :",
        manifest[
            "raw_state_tensor_count"
        ],
    )
    print(
        "EMA state tensors      :",
        manifest[
            "ema_state_tensor_count"
        ],
    )
    print(
        "Optimizer entries      :",
        manifest[
            "optimizer_state_entry_count"
        ],
    )

    save_quantum_checkpoint(
        signed,
        SIGNED_PATH,
    )

    reloaded = load_quantum_checkpoint(
        SIGNED_PATH,
        map_location="cpu",
    )

    reloaded_manifest = (
        verify_scientific_state_manifest(
            reloaded,
            required=True,
        )
    )

    assert reloaded_manifest is not None
    assert (
        reloaded_manifest["digest"]
        == manifest["digest"]
    )

    print("Signed reload verified : True")

    # Change one FP64 quantum value while preserving the old manifest.
    tampered = copy.deepcopy(reloaded)

    quantum_weights = tampered[
        "model_state_dict"
    ]["gate.quantum_weights"]

    original_value = float(
        quantum_weights.reshape(-1)[0]
    )

    quantum_weights.reshape(-1)[0] += (
        torch.tensor(
            1.0e-8,
            dtype=quantum_weights.dtype,
        )
    )

    changed_value = float(
        quantum_weights.reshape(-1)[0]
    )

    print(
        "Tampered tensor        :",
        "model_state_dict.gate.quantum_weights[0]",
    )
    print(
        "Original value         :",
        original_value,
    )
    print(
        "Changed value          :",
        changed_value,
    )

    # Use torch.save directly so the intentionally invalid payload can
    # be written without any future save-time guard.
    TAMPERED_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    torch.save(
        tampered,
        TAMPERED_PATH,
    )

    verification_failed = False
    failure_message = ""

    try:
        load_quantum_checkpoint(
            TAMPERED_PATH,
            map_location="cpu",
        )
    except ValueError as error:
        verification_failed = True
        failure_message = str(error)

    print(
        "Tamper detected        :",
        verification_failed,
    )
    print(
        "Failure message        :",
        failure_message,
    )

    assert verification_failed
    assert (
        "Scientific-state integrity "
        "verification failed"
        in failure_message
    )

    print("Signed checkpoint      :", SIGNED_PATH)
    print("Tampered checkpoint    :", TAMPERED_PATH)
    print()
    print(
        "SCIENTIFIC CHECKPOINT "
        "INTEGRITY: PASSED"
    )


if __name__ == "__main__":
    main()
