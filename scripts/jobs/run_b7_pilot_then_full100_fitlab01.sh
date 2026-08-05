#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="/export/users/1165521/iDragonCloud/QuantumC"
cd "$ROOT"

source scripts/activate_qvisionframe.sh

SERVER="${QVF_SERVER:-UNKNOWN}"

if [[ "$SERVER" != "FITLAB-01" ]]; then
  echo "ERROR: This pipeline is restricted to FITLAB-01." >&2
  echo "Detected QVF_SERVER=$SERVER" >&2
  exit 10
fi

if [[ "${QVF_COMPUTE_MODE:-unknown}" != "cuda" ]]; then
  echo "ERROR: CUDA compute mode is required." >&2
  echo "Detected QVF_COMPUTE_MODE=${QVF_COMPUTE_MODE:-unset}" >&2
  exit 11
fi

mkdir -p   "$ROOT/outputs/logs/$SERVER"   "$ROOT/outputs/pids"

PILOT_RUN="voc_b7_seed42_pilot5_b32"
FULL_RUN="voc_b7_seed42_full100_b32"

PILOT_DIR="$ROOT/outputs/$SERVER/voc_full_training/$PILOT_RUN"
FULL_DIR="$ROOT/outputs/$SERVER/voc_full_training/$FULL_RUN"

PILOT_LOG="$ROOT/outputs/logs/$SERVER/train_voc_b7_seed42_pilot5_b32.inner.log"
FULL_LOG="$ROOT/outputs/logs/$SERVER/train_voc_b7_seed42_full100_b32.inner.log"
STATUS_FILE="$ROOT/outputs/logs/$SERVER/b7_pipeline_status.txt"

timestamp() {
  date '+%Y-%m-%d %H:%M:%S %z'
}

status() {
  printf '[%s] %s\n' "$(timestamp)" "$*" | tee -a "$STATUS_FILE"
}

failure_handler() {
  exit_code=$?
  status "PIPELINE FAILED: exit_code=$exit_code line=${BASH_LINENO[0]}"
  exit "$exit_code"
}

trap failure_handler ERR

: > "$STATUS_FILE"

status "B7 PIPELINE STARTED"
status "Server: $SERVER"
status "Host: $(hostname)"
status "Pilot output: $PILOT_DIR"
status "Full output: $FULL_DIR"

python - <<'PY'
import torch

print("===== RUNTIME CHECK =====")
print("CUDA available:", torch.cuda.is_available())

if not torch.cuda.is_available():
    raise RuntimeError("CUDA is unavailable")

print("GPU:", torch.cuda.get_device_name(0))
print("RUNTIME CHECK: PASSED")
PY

# Không ghi đè một run đã tồn tại.
if [[ -e "$PILOT_DIR" ]]; then
  status "ERROR: Pilot directory already exists: $PILOT_DIR"
  exit 20
fi

if [[ -e "$FULL_DIR" ]]; then
  status "ERROR: Full-run directory already exists: $FULL_DIR"
  exit 21
fi

status "PHASE 1/3: Starting B7 pilot for 5 epochs"

python -u scripts/train_voc_full.py \
  --baseline B7 \
  --epochs 5 \
  --imgsz 640 \
  --batch 32 \
  --nbs 64 \
  --workers 8 \
  --cache false \
  --device 0 \
  --seed 42 \
  --save-period 1 \
  --run-suffix pilot5_b32 \
  2>&1 | tee "$PILOT_LOG"

PILOT_CHECKPOINT="$PILOT_DIR/weights/last.pt"

status "PHASE 1/3: Pilot process completed"
status "Expected checkpoint: $PILOT_CHECKPOINT"

if [[ ! -f "$PILOT_CHECKPOINT" ]]; then
  status "ERROR: Pilot checkpoint was not created"
  exit 30
fi

status "PHASE 2/3: Auditing pilot checkpoint"

PILOT_CHECKPOINT="$PILOT_CHECKPOINT" python - <<'PY'
from __future__ import annotations

import os
from pathlib import Path

import torch

from src.qvisionframe.quantum_checkpoint import (
    load_quantum_checkpoint,
)
from src.qvisionframe.scientific_state_hash import (
    scientific_state_sha256,
    verify_scientific_state_manifest,
)

path = Path(os.environ["PILOT_CHECKPOINT"]).resolve()

payload = load_quantum_checkpoint(
    path,
    map_location="cpu",
)

manifest = verify_scientific_state_manifest(
    payload,
    required=True,
)

baseline = payload[
    "model_metadata"
]["gate"]["baseline_id"]

entanglement = payload[
    "model_metadata"
]["gate"]["entanglement"]

epoch = int(payload["epoch"])
ema_updates = int(payload["ema_updates"])

optimizer_entries = len(
    payload["optimizer_state_dict"]["state"]
)

quantum_weights = payload[
    "model_state_dict"
]["gate.quantum_weights"]

computed_digest = scientific_state_sha256(
    payload
)

print("===== B7 PILOT CHECKPOINT AUDIT =====")
print("Checkpoint       :", path)
print("Baseline         :", baseline)
print("Entanglement     :", entanglement)
print("Epoch            :", epoch)
print("EMA updates      :", ema_updates)
print("Optimizer entries:", optimizer_entries)
print("Quantum dtype    :", quantum_weights.dtype)
print("Fitness          :", payload["best_fitness"])
print("Stored digest    :", manifest["digest"])
print("Computed digest  :", computed_digest)

assert baseline == "B7"
assert entanglement == "none"
assert epoch == 4
assert ema_updates > 0
assert optimizer_entries == 263
assert quantum_weights.dtype == torch.float64
assert torch.isfinite(quantum_weights).all()
assert manifest["digest"] == computed_digest

print("B7 PILOT CHECKPOINT AUDIT: PASSED")
PY

status "PHASE 2/3: Pilot checkpoint audit passed"
status "PHASE 3/3: Resuming B7 to 100 total epochs"

python -u scripts/train_voc_full.py \
  --baseline B7 \
  --epochs 100 \
  --imgsz 640 \
  --batch 32 \
  --nbs 64 \
  --workers 8 \
  --cache false \
  --device 0 \
  --seed 42 \
  --save-period 5 \
  --run-suffix full100_b32 \
  --resume-from "$PILOT_CHECKPOINT" \
  2>&1 | tee "$FULL_LOG"

FULL_CHECKPOINT="$FULL_DIR/weights/last.pt"

if [[ ! -f "$FULL_CHECKPOINT" ]]; then
  status "ERROR: Full-run checkpoint was not created"
  exit 40
fi

status "B7 FULL100 COMPLETED"
status "Final checkpoint: $FULL_CHECKPOINT"
status "PIPELINE PASSED"
