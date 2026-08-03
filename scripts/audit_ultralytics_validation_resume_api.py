from __future__ import annotations

import inspect
import json
from pathlib import Path
from typing import Any

import ultralytics
from ultralytics.engine.trainer import BaseTrainer
from ultralytics.engine.validator import BaseValidator
from ultralytics.models.yolo.detect import (
    DetectionTrainer,
    DetectionValidator,
)


TARGETS: list[tuple[str, Any]] = [
    (
        "BaseTrainer.validate",
        BaseTrainer.validate,
    ),
    (
        "BaseTrainer._load_checkpoint_state",
        BaseTrainer._load_checkpoint_state,
    ),
    (
        "BaseTrainer.setup_model",
        BaseTrainer.setup_model,
    ),
    (
        "BaseTrainer.check_resume",
        BaseTrainer.check_resume,
    ),
    (
        "DetectionTrainer.get_validator",
        DetectionTrainer.get_validator,
    ),
    (
        "BaseValidator.__call__",
        BaseValidator.__call__,
    ),
    (
        "DetectionValidator.__call__",
        DetectionValidator.__call__,
    ),
]


def source_record(
    qualified_name: str,
    value: Any,
) -> dict[str, Any]:
    try:
        signature = str(
            inspect.signature(value)
        )
    except (TypeError, ValueError):
        signature = "<unavailable>"

    try:
        source = inspect.getsource(value)
        source_file = inspect.getsourcefile(value)
        start_line = inspect.getsourcelines(value)[1]
    except (OSError, TypeError):
        source = "<source unavailable>"
        source_file = None
        start_line = None

    return {
        "qualified_name": qualified_name,
        "signature": signature,
        "source_file": source_file,
        "start_line": start_line,
        "source": source,
    }


def main() -> None:
    print(
        "===== ULTRALYTICS VALIDATION/RESUME API AUDIT ====="
    )
    print(
        "Ultralytics version:",
        ultralytics.__version__,
    )

    records: list[dict[str, Any]] = []

    for qualified_name, value in TARGETS:
        record = source_record(
            qualified_name,
            value,
        )
        records.append(record)

        print()
        print("=" * 78)
        print(record["qualified_name"])
        print("=" * 78)
        print("Signature  :", record["signature"])
        print("Source file:", record["source_file"])
        print("Start line :", record["start_line"])
        print()
        print(record["source"])

    validator_source = next(
        record["source"]
        for record in records
        if record["qualified_name"]
        == "BaseValidator.__call__"
    )

    load_state_source = next(
        record["source"]
        for record in records
        if record["qualified_name"]
        == "BaseTrainer._load_checkpoint_state"
    )

    tokens = {
        "validator_accepts_trainer": (
            "trainer" in validator_source
        ),
        "validator_accepts_model": (
            "model" in validator_source
        ),
        "validator_uses_autobackend": (
            "AutoBackend" in validator_source
        ),
        "loads_optimizer": (
            "optimizer" in load_state_source
        ),
        "loads_scaler": (
            "scaler" in load_state_source
        ),
        "loads_ema": (
            "ema" in load_state_source.lower()
        ),
    }

    print()
    print("===== CONTRACT TOKENS =====")

    for key, value in tokens.items():
        print(f"{key:<32}: {value}")

    root = Path(__file__).resolve().parents[1]

    output = {
        "ultralytics_version": (
            ultralytics.__version__
        ),
        "contract_tokens": tokens,
        "records": records,
    }

    output_path = (
        root
        / "outputs/inventory/"
        "ultralytics_validation_resume_api.json"
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path.write_text(
        json.dumps(output, indent=2) + "\n",
        encoding="utf-8",
    )

    assert ultralytics.__version__ == "8.4.102"
    assert "trainer" in validator_source
    assert "optimizer" in load_state_source

    print()
    print("Audit report:", output_path)
    print(
        "ULTRALYTICS VALIDATION/RESUME API AUDIT: PASSED"
    )


if __name__ == "__main__":
    main()
