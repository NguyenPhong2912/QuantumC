from __future__ import annotations

import json
from pathlib import Path

import torch

from src.qvisionframe.quantum_checkpoint import (
    load_quantum_checkpoint,
)
from src.qvisionframe.scientific_state_hash import (
    scientific_state_sha256,
    verify_scientific_state_manifest,
)


ROOT = Path(__file__).resolve().parents[1]

SERVER = "FITLAB-02"

EXPECTED = {
    "B6": {
        "entanglement": "ring",
        "trainable": False,
        "optimizer_entries": 262,
        "removed_parameter_tensors": 1,
        "delta_relation": "zero",
    },
    "B7": {
        "entanglement": "none",
        "trainable": True,
        "optimizer_entries": 263,
        "removed_parameter_tensors": 0,
        "delta_relation": "positive",
    },
    "B8": {
        "entanglement": "ring",
        "trainable": True,
        "optimizer_entries": 263,
        "removed_parameter_tensors": 0,
        "delta_relation": "positive",
    },
}

REPORT_PATH = (
    ROOT
    / "outputs/inventory/"
    "quantum_production_matrix.json"
)


def parse_summary(baseline: str) -> dict:
    path = (
        ROOT
        / f"outputs/{SERVER}/smoke_training/"
        f"voc_{baseline.lower()}_quantum_smoke_seed42/"
        f"{baseline.lower()}_quantum_smoke_summary.json"
    )

    if not path.exists():
        raise FileNotFoundError(path)

    return json.loads(
        path.read_text(encoding="utf-8")
    )


def main() -> None:
    print(
        "===== QUANTUM PRODUCTION MATRIX AUDIT ====="
    )

    report = {
        "server": SERVER,
        "baselines": {},
    }

    digests = set()

    for baseline, expected in EXPECTED.items():
        print()
        print(f"===== {baseline} =====")

        run_dir = (
            ROOT
            / f"outputs/{SERVER}/smoke_training/"
            f"voc_{baseline.lower()}_quantum_smoke_seed42"
        )

        weights_dir = run_dir / "weights"
        summary = parse_summary(baseline)

        max_abs_delta = float(
            summary["quantum_max_abs_delta"]
        )
        l2_delta = float(
            summary["quantum_l2_delta"]
        )

        raw_summary = summary["raw_state"]
        ema_summary = summary["ema_state"]

        print("Summary baseline      :", summary["baseline"])
        print("Summary entanglement  :", summary["entanglement"])
        print(
            "Summary optimizer     :",
            summary["optimizer_state_entries"],
        )
        print("Quantum max abs delta :", max_abs_delta)
        print("Quantum L2 delta      :", l2_delta)
        print("Raw summary           :", raw_summary)
        print("EMA summary           :", ema_summary)
        print(
            "Reload quantum dtype  :",
            summary["reload_quantum_dtype"],
        )
        print(
            "Reload feature shape  :",
            summary["reload_feature_shape"],
        )

        # Validate the schema actually emitted by
        # smoke_train_voc_quantum.py.
        assert summary["server"] == SERVER
        assert summary["baseline"] == baseline
        assert (
            summary["entanglement"]
            == expected["entanglement"]
        )
        assert summary["epoch"] == 0
        assert summary["ema_updates"] == 16
        assert (
            summary["optimizer_state_entries"]
            == expected["optimizer_entries"]
        )

        for state_summary in (
            raw_summary,
            ema_summary,
        ):
            assert state_summary["state_tensors"] == 507
            assert state_summary["gate_state_keys"] == 8
            assert (
                state_summary["quantum_dtype"]
                == "torch.float64"
            )
            assert state_summary["quantum_device"] == "cpu"
            assert state_summary["finite"] is True

        assert (
            summary["reload_quantum_dtype"]
            == "torch.float64"
        )
        assert summary["reload_hook_calls"] == 1
        assert (
            summary["reload_feature_shape"]
            == [1, 256, 10, 10]
        )

        if expected["delta_relation"] == "zero":
            assert max_abs_delta == 0.0
            assert l2_delta == 0.0
        else:
            assert torch.isfinite(
                torch.tensor(max_abs_delta)
            )
            assert torch.isfinite(
                torch.tensor(l2_delta)
            )
            assert max_abs_delta > 0.0
            assert l2_delta > 0.0

        checkpoint_records = {}

        for filename in ("last.pt", "best.pt"):
            path = weights_dir / filename

            assert path.exists(), path

            payload = load_quantum_checkpoint(
                path,
                map_location="cpu",
            )

            manifest = (
                verify_scientific_state_manifest(
                    payload,
                    required=True,
                )
            )

            assert manifest is not None

            computed = scientific_state_sha256(
                payload
            )

            gate = payload[
                "model_metadata"
            ]["gate"]

            raw_quantum = payload[
                "model_state_dict"
            ]["gate.quantum_weights"]

            ema_quantum = payload[
                "ema_state_dict"
            ]["gate.quantum_weights"]

            optimizer_state = payload[
                "optimizer_state_dict"
            ]

            print()
            print("Checkpoint :", filename)
            print("Baseline   :", gate["baseline_id"])
            print("Entangle   :", gate["entanglement"])
            print("Digest OK  :", manifest["digest"] == computed)
            print("Epoch      :", manifest["epoch"])
            print("EMA updates:", manifest["ema_updates"])
            print(
                "Optimizer  :",
                manifest[
                    "optimizer_state_entry_count"
                ],
            )
            print(
                "Raw quantum:",
                raw_quantum.dtype,
            )
            print(
                "EMA quantum:",
                ema_quantum.dtype,
            )

            assert gate["baseline_id"] == baseline
            assert (
                gate["entanglement"]
                == expected["entanglement"]
            )
            assert manifest["digest"] == computed
            assert manifest["epoch"] == 0
            assert manifest["ema_updates"] == 16
            assert (
                manifest[
                    "raw_state_tensor_count"
                ]
                == 507
            )
            assert (
                manifest[
                    "ema_state_tensor_count"
                ]
                == 507
            )
            assert (
                manifest[
                    "optimizer_state_entry_count"
                ]
                == expected["optimizer_entries"]
            )
            assert (
                len(optimizer_state["state"])
                == expected["optimizer_entries"]
            )
            assert raw_quantum.dtype == torch.float64
            assert ema_quantum.dtype == torch.float64
            assert bool(torch.isfinite(raw_quantum).all())
            assert bool(torch.isfinite(ema_quantum).all())

            checkpoint_records[filename] = {
                "digest": manifest["digest"],
                "epoch": manifest["epoch"],
                "ema_updates": manifest[
                    "ema_updates"
                ],
                "optimizer_entries": manifest[
                    "optimizer_state_entry_count"
                ],
            }

            digests.add(manifest["digest"])

        assert (
            checkpoint_records["last.pt"]["digest"]
            == checkpoint_records["best.pt"]["digest"]
        )

        report["baselines"][baseline] = {
            "entanglement": expected[
                "entanglement"
            ],
            "trainable": expected["trainable"],
            "optimizer_entries": expected[
                "optimizer_entries"
            ],
            "summary_contract": {
                "server": summary["server"],
                "baseline": summary["baseline"],
                "entanglement": summary["entanglement"],
                "epoch": summary["epoch"],
                "ema_updates": summary["ema_updates"],
                "optimizer_state_entries": summary[
                    "optimizer_state_entries"
                ],
                "raw_state": raw_summary,
                "ema_state": ema_summary,
                "reload_quantum_dtype": summary[
                    "reload_quantum_dtype"
                ],
                "reload_hook_calls": summary[
                    "reload_hook_calls"
                ],
                "reload_feature_shape": summary[
                    "reload_feature_shape"
                ],
            },
            "quantum_max_abs_delta": max_abs_delta,
            "quantum_l2_delta": l2_delta,
            "checkpoints": checkpoint_records,
            "last_best_exact": True,
        }

    # Different ablations must not collapse to one scientific state.
    assert len(digests) == 3

    REPORT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    REPORT_PATH.write_text(
        json.dumps(
            report,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    print()
    print("Distinct scientific states:", len(digests))
    print("Report                    :", REPORT_PATH)
    print()
    print(
        "QUANTUM PRODUCTION MATRIX AUDIT: PASSED"
    )


if __name__ == "__main__":
    main()
