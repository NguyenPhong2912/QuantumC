from __future__ import annotations

import inspect
import json
from pathlib import Path
from typing import Any

import ultralytics
from ultralytics.engine.trainer import BaseTrainer
from ultralytics.utils import torch_utils
from ultralytics.utils.torch_utils import ModelEMA


TARGETS: list[tuple[str, Any]] = [
    ("BaseTrainer.save_model", BaseTrainer.save_model),
    ("BaseTrainer._do_train", BaseTrainer._do_train),
    ("BaseTrainer.final_eval", BaseTrainer.final_eval),
    ("BaseTrainer.resume_training", BaseTrainer.resume_training),
    ("ModelEMA.__init__", ModelEMA.__init__),
    ("ModelEMA.update", ModelEMA.update),
]


OPTIONAL_TARGETS = (
    "strip_optimizer",
    "convert_optimizer_state_dict_to_fp16",
)


def source_record(
    qualified_name: str,
    value: Any,
) -> dict[str, Any]:
    try:
        signature = str(inspect.signature(value))
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


def print_record(
    record: dict[str, Any],
) -> None:
    print()
    print("=" * 78)
    print(record["qualified_name"])
    print("=" * 78)
    print("Signature  :", record["signature"])
    print("Source file:", record["source_file"])
    print("Start line :", record["start_line"])
    print()
    print(record["source"])


def main() -> None:
    print("===== ULTRALYTICS CHECKPOINT API AUDIT =====")
    print("Ultralytics version:", ultralytics.__version__)
    print("Package path       :", Path(ultralytics.__file__).resolve())

    records: list[dict[str, Any]] = []

    for qualified_name, value in TARGETS:
        record = source_record(
            qualified_name,
            value,
        )
        records.append(record)
        print_record(record)

    for name in OPTIONAL_TARGETS:
        value = getattr(
            torch_utils,
            name,
            None,
        )

        if value is None:
            print()
            print(
                f"OPTIONAL TARGET MISSING: "
                f"ultralytics.utils.torch_utils.{name}"
            )
            continue

        record = source_record(
            f"ultralytics.utils.torch_utils.{name}",
            value,
        )
        records.append(record)
        print_record(record)

    trainer_source = inspect.getsource(
        BaseTrainer._do_train
    )

    lifecycle_tokens = {
        token: trainer_source.find(token)
        for token in (
            "ModelEMA",
            "self.ema",
            "self.save_model",
            "on_model_save",
            "self.final_eval",
            "on_train_end",
        )
    }

    print()
    print("===== TRAINING LIFECYCLE TOKEN POSITIONS =====")

    for token, position in lifecycle_tokens.items():
        print(f"{token:<24}: {position}")

    output = {
        "ultralytics_version": ultralytics.__version__,
        "package_path": str(
            Path(ultralytics.__file__).resolve()
        ),
        "lifecycle_token_positions": lifecycle_tokens,
        "records": records,
    }

    root = Path(__file__).resolve().parents[1]

    output_path = (
        root
        / "outputs/inventory/"
        "ultralytics_checkpoint_api.json"
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path.write_text(
        json.dumps(output, indent=2) + "\n",
        encoding="utf-8",
    )

    save_source = next(
        record["source"]
        for record in records
        if record["qualified_name"]
        == "BaseTrainer.save_model"
    )

    assert ultralytics.__version__ == "8.4.102"
    assert "torch.save" in save_source
    assert "optimizer" in save_source
    assert "ema" in save_source.lower()

    print()
    print("Audit report:", output_path)
    print("ULTRALYTICS CHECKPOINT API AUDIT: PASSED")


if __name__ == "__main__":
    main()
