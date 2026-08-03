from __future__ import annotations

import hashlib
import hmac
import json
import math
import struct
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import torch


SCIENTIFIC_HASH_SCHEMA = "qvisionframe.scientific-state.v1"


# These fields describe execution location or wall-clock performance.
# They must not affect scientific-state equivalence.
EXCLUDED_TRAIN_ARG_KEYS = {
    "project",
    "name",
    "save_dir",
    "resume",
    "device",
    "workers",
}

EXCLUDED_TRAIN_RESULT_KEYS = {
    "time",
}


def _write_bytes(
    digest: hashlib._Hash,
    value: bytes,
) -> None:
    digest.update(
        struct.pack(">Q", len(value))
    )
    digest.update(value)


def _write_text(
    digest: hashlib._Hash,
    value: str,
) -> None:
    _write_bytes(
        digest,
        value.encode("utf-8"),
    )


def _canonical_float_text(
    value: float,
) -> str:
    if math.isnan(value):
        return "nan"

    if math.isinf(value):
        return (
            "+inf"
            if value > 0
            else "-inf"
        )

    # Hex representation is exact and locale independent.
    return value.hex()


def _tensor_bytes(
    tensor: torch.Tensor,
) -> bytes:
    tensor = (
        tensor.detach()
        .cpu()
        .contiguous()
    )

    if tensor.layout != torch.strided:
        raise TypeError(
            "Only strided tensors are supported, got "
            f"{tensor.layout}"
        )

    # NumPy does not expose bfloat16 directly. Reinterpret its storage
    # through uint16 while keeping dtype metadata in the hash.
    if tensor.dtype == torch.bfloat16:
        return (
            tensor.view(torch.uint16)
            .numpy()
            .tobytes(order="C")
        )

    # Complex, integer, boolean and standard floating dtypes are
    # represented by their contiguous CPU storage.
    return tensor.numpy().tobytes(
        order="C"
    )


def update_canonical_hash(
    digest: hashlib._Hash,
    value: Any,
) -> None:
    """
    Add a Python/tensor value to a deterministic SHA-256 stream.

    Every value is type-tagged. Mapping keys are sorted by their
    canonical string form, making the result independent of dictionary
    insertion order.
    """
    if value is None:
        _write_text(digest, "none")
        return

    if isinstance(value, bool):
        _write_text(
            digest,
            "bool:1" if value else "bool:0",
        )
        return

    if isinstance(value, int):
        _write_text(
            digest,
            f"int:{value}",
        )
        return

    if isinstance(value, float):
        _write_text(
            digest,
            "float:"
            + _canonical_float_text(value),
        )
        return

    if isinstance(value, str):
        _write_text(digest, "str")
        _write_text(digest, value)
        return

    if isinstance(value, Path):
        _write_text(digest, "path")
        _write_text(digest, str(value))
        return

    if isinstance(value, torch.dtype):
        _write_text(
            digest,
            f"torch-dtype:{value}",
        )
        return

    if isinstance(value, torch.device):
        _write_text(
            digest,
            f"torch-device:{value}",
        )
        return

    if isinstance(value, np.generic):
        update_canonical_hash(
            digest,
            value.item(),
        )
        return

    if isinstance(value, torch.Tensor):
        _write_text(digest, "tensor")
        _write_text(
            digest,
            str(value.dtype),
        )
        _write_text(
            digest,
            str(value.layout),
        )

        update_canonical_hash(
            digest,
            list(value.shape),
        )

        _write_bytes(
            digest,
            _tensor_bytes(value),
        )
        return

    if isinstance(value, Mapping):
        _write_text(digest, "mapping")

        sorted_items = sorted(
            value.items(),
            key=lambda item: (
                type(item[0]).__name__,
                repr(item[0]),
            ),
        )

        update_canonical_hash(
            digest,
            len(sorted_items),
        )

        for key, item in sorted_items:
            update_canonical_hash(
                digest,
                key,
            )
            update_canonical_hash(
                digest,
                item,
            )

        return

    if isinstance(value, tuple):
        _write_text(digest, "tuple")
        update_canonical_hash(
            digest,
            len(value),
        )

        for item in value:
            update_canonical_hash(
                digest,
                item,
            )

        return

    if isinstance(value, list):
        _write_text(digest, "list")
        update_canonical_hash(
            digest,
            len(value),
        )

        for item in value:
            update_canonical_hash(
                digest,
                item,
            )

        return

    raise TypeError(
        "Unsupported canonical-hash value: "
        f"{type(value).__module__}."
        f"{type(value).__qualname__}"
    )


def stable_train_args(
    train_args: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        key: value
        for key, value in train_args.items()
        if key not in EXCLUDED_TRAIN_ARG_KEYS
    }


def stable_train_results(
    train_results: Any,
) -> Any:
    if not isinstance(
        train_results,
        Mapping,
    ):
        return train_results

    return {
        key: value
        for key, value in train_results.items()
        if key not in EXCLUDED_TRAIN_RESULT_KEYS
    }


def scientific_state_view(
    checkpoint: Mapping[str, Any],
) -> dict[str, Any]:
    """
    Return the stable portion of a Q-VisionFrame checkpoint.

    Operational metadata is excluded, but optimizer parameter groups
    and all tensor states are retained.
    """
    return {
        "schema": SCIENTIFIC_HASH_SCHEMA,
        "qvf_format": checkpoint[
            "qvf_format"
        ],
        "format_version": checkpoint[
            "format_version"
        ],
        "epoch": checkpoint["epoch"],
        "best_fitness": checkpoint.get(
            "best_fitness"
        ),
        "model_metadata": checkpoint[
            "model_metadata"
        ],
        "model_state_dict": checkpoint[
            "model_state_dict"
        ],
        "ema_state_dict": checkpoint[
            "ema_state_dict"
        ],
        "ema_updates": checkpoint[
            "ema_updates"
        ],
        "optimizer_state_dict": checkpoint.get(
            "optimizer_state_dict"
        ),
        "scaler_state_dict": checkpoint.get(
            "scaler_state_dict"
        ),
        "train_args": stable_train_args(
            checkpoint.get(
                "train_args",
                {},
            )
        ),
        "train_metrics": checkpoint.get(
            "train_metrics",
            {},
        ),
        "train_results": stable_train_results(
            checkpoint.get(
                "train_results"
            )
        ),
    }


def scientific_state_sha256(
    checkpoint: Mapping[str, Any],
) -> str:
    digest = hashlib.sha256()

    update_canonical_hash(
        digest,
        scientific_state_view(checkpoint),
    )

    return digest.hexdigest()


def scientific_hash_manifest(
    checkpoint: Mapping[str, Any],
) -> dict[str, Any]:
    view = scientific_state_view(
        checkpoint
    )

    return {
        "schema": SCIENTIFIC_HASH_SCHEMA,
        "algorithm": "sha256",
        "digest": scientific_state_sha256(
            checkpoint
        ),
        "epoch": view["epoch"],
        "ema_updates": view["ema_updates"],
        "baseline_id": view[
            "model_metadata"
        ]["gate"]["baseline_id"],
        "entanglement": view[
            "model_metadata"
        ]["gate"]["entanglement"],
        "raw_state_tensor_count": len(
            view["model_state_dict"]
        ),
        "ema_state_tensor_count": len(
            view["ema_state_dict"]
        ),
        "optimizer_state_entry_count": (
            0
            if view[
                "optimizer_state_dict"
            ]
            is None
            else len(
                view[
                    "optimizer_state_dict"
                ]["state"]
            )
        ),
        "excluded_train_arg_keys": sorted(
            EXCLUDED_TRAIN_ARG_KEYS
        ),
        "excluded_train_result_keys": sorted(
            EXCLUDED_TRAIN_RESULT_KEYS
        ),
    }


def write_scientific_hash_manifest(
    checkpoint: Mapping[str, Any],
    output_path: str | Path,
) -> Path:
    path = Path(output_path)
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    path.write_text(
        json.dumps(
            scientific_hash_manifest(
                checkpoint
            ),
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    return path


def attach_scientific_state_manifest(
    checkpoint: dict[str, Any],
) -> dict[str, Any]:
    """
    Attach a canonical scientific-state manifest in place.

    scientific_state_view() deliberately ignores this manifest, so
    computing the digest does not create a recursive hash dependency.
    """
    checkpoint["scientific_state"] = (
        scientific_hash_manifest(checkpoint)
    )

    return checkpoint


def verify_scientific_state_manifest(
    checkpoint: Mapping[str, Any],
    *,
    required: bool = False,
) -> dict[str, Any] | None:
    """
    Verify an embedded scientific-state manifest.

    Args:
        checkpoint:
            Q-VisionFrame checkpoint payload.
        required:
            Reject legacy checkpoints without an embedded manifest.

    Returns:
        A copy of the verified manifest, or None for a legacy
        checkpoint when required=False.
    """
    manifest = checkpoint.get(
        "scientific_state"
    )

    if manifest is None:
        if required:
            raise ValueError(
                "Checkpoint has no embedded scientific-state manifest"
            )

        return None

    if not isinstance(manifest, Mapping):
        raise TypeError(
            "scientific_state must be a mapping"
        )

    schema = manifest.get("schema")
    algorithm = manifest.get("algorithm")
    stored_digest = manifest.get("digest")

    if schema != SCIENTIFIC_HASH_SCHEMA:
        raise ValueError(
            "Unsupported scientific-state schema: "
            f"{schema!r}"
        )

    if algorithm != "sha256":
        raise ValueError(
            "Unsupported scientific-state hash algorithm: "
            f"{algorithm!r}"
        )

    if (
        not isinstance(stored_digest, str)
        or len(stored_digest) != 64
    ):
        raise ValueError(
            "Invalid embedded scientific-state digest"
        )

    computed_digest = scientific_state_sha256(
        checkpoint
    )

    if not hmac.compare_digest(
        stored_digest,
        computed_digest,
    ):
        raise ValueError(
            "Scientific-state integrity verification failed: "
            f"stored={stored_digest}, "
            f"computed={computed_digest}"
        )

    expected_inventory = {
        "epoch": checkpoint["epoch"],
        "ema_updates": checkpoint["ema_updates"],
        "baseline_id": checkpoint[
            "model_metadata"
        ]["gate"]["baseline_id"],
        "entanglement": checkpoint[
            "model_metadata"
        ]["gate"]["entanglement"],
        "raw_state_tensor_count": len(
            checkpoint["model_state_dict"]
        ),
        "ema_state_tensor_count": len(
            checkpoint["ema_state_dict"]
        ),
        "optimizer_state_entry_count": (
            0
            if checkpoint.get(
                "optimizer_state_dict"
            )
            is None
            else len(
                checkpoint[
                    "optimizer_state_dict"
                ]["state"]
            )
        ),
    }

    for key, expected in (
        expected_inventory.items()
    ):
        actual = manifest.get(key)

        if actual != expected:
            raise ValueError(
                "Scientific-state manifest inventory mismatch "
                f"for {key}: stored={actual!r}, "
                f"expected={expected!r}"
            )

    return dict(manifest)
