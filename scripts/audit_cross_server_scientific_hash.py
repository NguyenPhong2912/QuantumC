from __future__ import annotations

import json
from pathlib import Path

from src.qvisionframe.quantum_checkpoint import (
    load_quantum_checkpoint,
)
from src.qvisionframe.scientific_state_hash import (
    scientific_hash_manifest,
    write_scientific_hash_manifest,
)


ROOT = Path(__file__).resolve().parents[1]

CHECKPOINTS = {
    "FITLAB-01": (
        ROOT
        / "outputs/FITLAB-01/resume_training/"
        "voc_b8_quantum_resume_seed42/"
        "weights/last.pt"
    ),
    "FITLAB-02": (
        ROOT
        / "outputs/FITLAB-02/resume_training/"
        "voc_b8_quantum_resume_seed42/"
        "weights/last.pt"
    ),
}

REPORT_DIRECTORY = (
    ROOT
    / "outputs/inventory/"
    "scientific_hash"
)


def main() -> None:
    print(
        "===== CROSS-SERVER SCIENTIFIC "
        "STATE HASH AUDIT ====="
    )

    manifests = {}

    for server, checkpoint_path in (
        CHECKPOINTS.items()
    ):
        assert checkpoint_path.exists()

        checkpoint = load_quantum_checkpoint(
            checkpoint_path,
            map_location="cpu",
        )

        manifest = scientific_hash_manifest(
            checkpoint
        )

        manifests[server] = manifest

        output_path = (
            REPORT_DIRECTORY
            / f"{server.lower()}_"
            "b8_resume_scientific_hash.json"
        )

        write_scientific_hash_manifest(
            checkpoint,
            output_path,
        )

        print()
        print("Server             :", server)
        print(
            "Checkpoint         :",
            checkpoint_path,
        )
        print(
            "Scientific SHA256  :",
            manifest["digest"],
        )
        print(
            "Epoch              :",
            manifest["epoch"],
        )
        print(
            "EMA updates        :",
            manifest["ema_updates"],
        )
        print(
            "Raw state tensors  :",
            manifest[
                "raw_state_tensor_count"
            ],
        )
        print(
            "EMA state tensors  :",
            manifest[
                "ema_state_tensor_count"
            ],
        )
        print(
            "Optimizer entries  :",
            manifest[
                "optimizer_state_entry_count"
            ],
        )
        print("Manifest           :", output_path)

    digest_01 = manifests[
        "FITLAB-01"
    ]["digest"]

    digest_02 = manifests[
        "FITLAB-02"
    ]["digest"]

    print()
    print("===== HASH COMPARISON =====")
    print("FITLAB-01:", digest_01)
    print("FITLAB-02:", digest_02)
    print(
        "Scientific hashes exact:",
        digest_01 == digest_02,
    )

    assert digest_01 == digest_02
    assert (
        manifests["FITLAB-01"]
        ["raw_state_tensor_count"]
        == 507
    )
    assert (
        manifests["FITLAB-02"]
        ["raw_state_tensor_count"]
        == 507
    )
    assert (
        manifests["FITLAB-01"]
        ["ema_state_tensor_count"]
        == 507
    )
    assert (
        manifests["FITLAB-02"]
        ["ema_state_tensor_count"]
        == 507
    )
    assert (
        manifests["FITLAB-01"]
        ["optimizer_state_entry_count"]
        == 263
    )
    assert (
        manifests["FITLAB-02"]
        ["optimizer_state_entry_count"]
        == 263
    )

    report = {
        "fitlab01": manifests[
            "FITLAB-01"
        ],
        "fitlab02": manifests[
            "FITLAB-02"
        ],
        "scientific_hashes_exact": (
            digest_01 == digest_02
        ),
    }

    report_path = (
        REPORT_DIRECTORY
        / "cross_server_b8_"
        "scientific_hash_audit.json"
    )

    report_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    report_path.write_text(
        json.dumps(
            report,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    print("Report    :", report_path)
    print()
    print(
        "CROSS-SERVER SCIENTIFIC "
        "STATE HASH: PASSED"
    )


if __name__ == "__main__":
    main()
