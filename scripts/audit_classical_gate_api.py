from __future__ import annotations

import importlib
import inspect
import json
from dataclasses import fields, is_dataclass
from pathlib import Path
from types import ModuleType
from typing import Any

import torch


CANDIDATE_MODULES = (
    "src.qvisionframe.classical_gate",
    "src.qvisionframe.channel_attention_gate",
    "src.qvisionframe.parameter_matched_gate",
    "src.qvisionframe.trigonometric_gate",
    "src.qvisionframe.baselines",
    "src.qvisionframe.baseline_factory",
)


KEYWORDS = (
    "gate",
    "attention",
    "linear",
    "classical",
    "parameter",
    "trigonometric",
    "config",
)


def safe_import(module_name: str) -> ModuleType | None:
    try:
        return importlib.import_module(module_name)
    except ModuleNotFoundError as error:
        if error.name == module_name:
            return None
        raise


def class_details(
    module_name: str,
    name: str,
    value: type[Any],
) -> dict[str, Any]:
    try:
        signature = str(inspect.signature(value))
    except (TypeError, ValueError):
        signature = "<unavailable>"

    dataclass_fields: list[dict[str, Any]] = []

    if is_dataclass(value):
        for field in fields(value):
            default: Any

            if field.default is not inspect.Parameter.empty:
                default = repr(field.default)
            else:
                default = "<required>"

            dataclass_fields.append(
                {
                    "name": field.name,
                    "type": str(field.type),
                    "default": default,
                }
            )

    return {
        "module": module_name,
        "name": name,
        "qualified_name": f"{module_name}.{name}",
        "signature": signature,
        "is_nn_module": issubclass(value, torch.nn.Module),
        "is_dataclass": is_dataclass(value),
        "dataclass_fields": dataclass_fields,
    }


def main() -> None:
    print("===== CLASSICAL GATE API AUDIT =====")

    root = Path(__file__).resolve().parents[1]
    records: list[dict[str, Any]] = []
    imported_modules: list[str] = []
    missing_modules: list[str] = []

    for module_name in CANDIDATE_MODULES:
        module = safe_import(module_name)

        if module is None:
            missing_modules.append(module_name)
            continue

        imported_modules.append(module_name)

        print()
        print("MODULE:", module_name)

        public_names = sorted(
            name
            for name in vars(module)
            if not name.startswith("_")
        )

        relevant_names = [
            name
            for name in public_names
            if any(
                keyword in name.lower()
                for keyword in KEYWORDS
            )
        ]

        for name in relevant_names:
            value = getattr(module, name)

            if not inspect.isclass(value):
                continue

            record = class_details(
                module_name,
                name,
                value,
            )
            records.append(record)

            print(
                f"  {name:<42} "
                f"nn.Module={record['is_nn_module']!s:<5} "
                f"dataclass={record['is_dataclass']!s:<5}"
            )
            print(
                "    signature:",
                record["signature"],
            )

            for field in record["dataclass_fields"]:
                print(
                    "    field:",
                    field["name"],
                    "| type=",
                    field["type"],
                    "| default=",
                    field["default"],
                )

    output = {
        "imported_modules": imported_modules,
        "missing_modules": missing_modules,
        "classes": records,
    }

    output_path = (
        root
        / "outputs/inventory/"
        "classical_gate_api.json"
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path.write_text(
        json.dumps(output, indent=2) + "\n",
        encoding="utf-8",
    )

    nn_gate_classes = [
        record
        for record in records
        if record["is_nn_module"]
        and "gate" in record["name"].lower()
    ]

    config_classes = [
        record
        for record in records
        if record["is_dataclass"]
        and "config" in record["name"].lower()
    ]

    print()
    print("===== SUMMARY =====")
    print("Imported modules :", imported_modules)
    print("Missing modules  :", missing_modules)
    print("Gate classes     :", len(nn_gate_classes))
    print("Config classes   :", len(config_classes))
    print("Inventory        :", output_path)

    for record in nn_gate_classes:
        print(
            "GATE  :",
            record["qualified_name"],
            record["signature"],
        )

    for record in config_classes:
        print(
            "CONFIG:",
            record["qualified_name"],
            record["signature"],
        )

    assert imported_modules
    assert nn_gate_classes
    assert config_classes

    print("CLASSICAL GATE API AUDIT: PASSED")


if __name__ == "__main__":
    main()
