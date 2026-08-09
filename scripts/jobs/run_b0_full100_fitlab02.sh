#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(
  cd "$(dirname "${BASH_SOURCE[0]}")/../.."
  pwd
)"
cd "$ROOT"
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"

MODE="${1:---run}"

if [[ "$MODE" != "--run" &&
      "$MODE" != "--preflight-only" ]]; then
  echo "Usage: $0 [--run|--preflight-only]" >&2
  exit 2
fi

SERVER="${QVF_SERVER:-FITLAB-02}"
PYTHON="$ROOT/.venv-fitlab02/bin/python"
DATA_YAML="${QVF_DATA_YAML:-$ROOT/configs/datasets/voc.yaml}"
MIN_FREE_GIB="${QVF_MIN_FREE_GIB:-10}"
MIN_FREE_VRAM_MIB="${QVF_MIN_FREE_VRAM_MIB:-9000}"

if [[ "$SERVER" != "FITLAB-02" ]]; then
  echo "ERROR: B0 full100 is restricted to FITLAB-02." >&2
  echo "Detected QVF_SERVER=$SERVER" >&2
  exit 10
fi

if [[ ! -x "$PYTHON" ]]; then
  echo "ERROR: Python environment not found: $PYTHON" >&2
  exit 11
fi

if [[ ! -f "$DATA_YAML" ]]; then
  echo "ERROR: Dataset YAML not found: $DATA_YAML" >&2
  exit 12
fi

BRANCH="$(git branch --show-current)"

if [[ "$BRANCH" != "AnKy06" ]]; then
  echo "ERROR: Expected branch AnKy06, received $BRANCH" >&2
  exit 13
fi

if git show-ref --verify --quiet refs/remotes/origin/AnKy06; then
  LOCAL_COMMIT="$(git rev-parse HEAD)"
  REMOTE_COMMIT="$(git rev-parse origin/AnKy06)"

  if [[ "$LOCAL_COMMIT" != "$REMOTE_COMMIT" ]]; then
    echo "ERROR: Local AnKy06 is not synchronized with origin/AnKy06." >&2
    echo "Local : $LOCAL_COMMIT" >&2
    echo "Remote: $REMOTE_COMMIT" >&2
    exit 14
  fi
fi

unexpected_changes=()

while IFS= read -r status_line; do
  [[ -n "$status_line" ]] || continue
  path="${status_line:3}"

  case "$path" in
    AGENTS.md|configs/datasets/voc.yaml|configs/datasets/voc_smoke_32.yaml)
      ;;
    *)
      unexpected_changes+=("$status_line")
      ;;
  esac
done < <(git status --porcelain --untracked-files=no)

if [[ "${#unexpected_changes[@]}" -gt 0 ]]; then
  echo "ERROR: Unexpected tracked working-tree changes:" >&2
  printf '  %s\n' "${unexpected_changes[@]}" >&2
  exit 15
fi

DATA_ROOT="$($PYTHON - "$DATA_YAML" <<'PY'
from pathlib import Path
import sys
import yaml

yaml_path = Path(sys.argv[1]).resolve()
payload = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
root = Path(payload["path"]).expanduser()

if not root.is_absolute():
    root = (yaml_path.parent / root).resolve()

print(root)
PY
)"

if [[ ! -d "$DATA_ROOT" ]]; then
  echo "ERROR: Dataset root not found: $DATA_ROOT" >&2
  exit 16
fi

declare -A expected_counts=(
  [train]=16551
  [val]=4952
)

for split in train val; do
  image_count="$(
    find "$DATA_ROOT/images/$split" -maxdepth 1 -type f | wc -l
  )"
  label_count="$(
    find "$DATA_ROOT/labels/$split" -maxdepth 1 -type f | wc -l
  )"

  if [[ "$image_count" -ne "${expected_counts[$split]}" ||
        "$label_count" -ne "${expected_counts[$split]}" ]]; then
    echo "ERROR: VOC $split count mismatch." >&2
    echo "Images=$image_count labels=$label_count expected=${expected_counts[$split]}" >&2
    exit 17
  fi
done

AVAILABLE_KIB="$(df -Pk "$ROOT" | awk 'NR == 2 {print $4}')"
MIN_FREE_KIB="$((MIN_FREE_GIB * 1024 * 1024))"

if [[ -z "$AVAILABLE_KIB" || "$AVAILABLE_KIB" -lt "$MIN_FREE_KIB" ]]; then
  echo "ERROR: Insufficient filesystem space." >&2
  echo "Available KiB=${AVAILABLE_KIB:-unknown}; required KiB=$MIN_FREE_KIB" >&2
  exit 18
fi

FREE_VRAM_MIB="$(
  nvidia-smi \
    --query-gpu=memory.free \
    --format=csv,noheader,nounits |
  head -n 1 |
  tr -d ' '
)"

if [[ -z "$FREE_VRAM_MIB" ||
      "$FREE_VRAM_MIB" -lt "$MIN_FREE_VRAM_MIB" ]]; then
  echo "ERROR: Insufficient free GPU memory." >&2
  echo "Free MiB=${FREE_VRAM_MIB:-unknown}; required MiB=$MIN_FREE_VRAM_MIB" >&2
  exit 19
fi

$PYTHON - <<'PY'
import torch
import ultralytics

if not torch.cuda.is_available():
    raise RuntimeError("CUDA is unavailable")

print("CUDA available     :", torch.cuda.is_available())
print("GPU                :", torch.cuda.get_device_name(0))
print("Torch              :", torch.__version__)
print("Ultralytics        :", ultralytics.__version__)
PY

echo "===== B0 FULL100 PREFLIGHT ====="
echo "Root               : $ROOT"
echo "Branch             : $BRANCH"
echo "Commit             : $(git rev-parse HEAD)"
echo "Server             : $SERVER"
echo "Dataset YAML       : $DATA_YAML"
echo "Dataset root       : $DATA_ROOT"
echo "VOC train          : ${expected_counts[train]}"
echo "VOC val            : ${expected_counts[val]}"
echo "Filesystem free GiB: $((AVAILABLE_KIB / 1024 / 1024))"
echo "GPU free MiB       : $FREE_VRAM_MIB"
echo "B0 FULL100 PREFLIGHT: PASSED"

if [[ "$MODE" == "--preflight-only" ]]; then
  exit 0
fi

START_STAMP="$(date '+%Y%m%d_%H%M%S')"
START_TIME="$(date --iso-8601=seconds)"
RUN_SUFFIX="${QVF_RUN_SUFFIX:-full100_b32_${START_STAMP}}"
RUN_NAME="voc_b0_seed42_${RUN_SUFFIX}"
RUN_DIR="$ROOT/outputs/$SERVER/voc_full_training/$RUN_NAME"
LOG_DIR="$ROOT/outputs/logs/$SERVER"
PID_DIR="$ROOT/outputs/pids"
REGISTRY_DIR="$ROOT/outputs/registry/$SERVER"
INNER_LOG="$LOG_DIR/${RUN_NAME}.inner.log"
STATUS_FILE="$LOG_DIR/${RUN_NAME}.status.log"
PID_FILE="$PID_DIR/train_voc_b0_full100.pid"
REGISTRY_FILE="$REGISTRY_DIR/${RUN_NAME}.json"
CURRENT_REGISTRY="$REGISTRY_DIR/b0_current.json"

mkdir -p "$LOG_DIR" "$PID_DIR" "$REGISTRY_DIR"

if [[ -e "$RUN_DIR" ||
      -e "$INNER_LOG" ||
      -e "$REGISTRY_FILE" ]]; then
  echo "ERROR: Refusing to overwrite an existing B0 run." >&2
  echo "Run directory: $RUN_DIR" >&2
  exit 20
fi

TRAIN_COMMAND=(
  "$PYTHON" -u scripts/train_voc_full.py
  --baseline B0
  --data-yaml "$DATA_YAML"
  --epochs 100
  --imgsz 640
  --batch 32
  --nbs 64
  --workers 8
  --cache false
  --device 0
  --seed 42
  --save-period 5
  --run-suffix "$RUN_SUFFIX"
)

printf -v TRAIN_COMMAND_TEXT '%q ' "${TRAIN_COMMAND[@]}"

export ROOT SERVER DATA_YAML DATA_ROOT START_TIME RUN_NAME RUN_DIR
export INNER_LOG STATUS_FILE PID_FILE REGISTRY_FILE CURRENT_REGISTRY
export TRAIN_COMMAND_TEXT FREE_VRAM_MIB AVAILABLE_KIB

write_registry() {
  local registry_status="$1"
  local exit_code="${2:-}"
  local end_time="${3:-}"

  REGISTRY_STATUS="$registry_status" \
  REGISTRY_EXIT_CODE="$exit_code" \
  REGISTRY_END_TIME="$end_time" \
  "$PYTHON" - <<'PY'
from __future__ import annotations

import hashlib
import json
import os
import socket
import subprocess
from pathlib import Path


def sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


registry_path = Path(os.environ["REGISTRY_FILE"])
current_path = Path(os.environ["CURRENT_REGISTRY"])
data_yaml = Path(os.environ["DATA_YAML"])
integrity_summary = (
    Path(os.environ["DATA_ROOT"])
    / "manifests/integrity_summary.csv"
)

payload = {
    "run_name": os.environ["RUN_NAME"],
    "baseline": "B0",
    "seed": 42,
    "status": os.environ["REGISTRY_STATUS"],
    "start_time": os.environ["START_TIME"],
    "end_time": os.environ.get("REGISTRY_END_TIME") or None,
    "exit_code": (
        int(os.environ["REGISTRY_EXIT_CODE"])
        if os.environ.get("REGISTRY_EXIT_CODE")
        else None
    ),
    "host": socket.gethostname(),
    "server": os.environ["SERVER"],
    "pid": int(Path(os.environ["PID_FILE"]).read_text().strip()),
    "git_commit": subprocess.check_output(
        ["git", "rev-parse", "HEAD"], text=True
    ).strip(),
    "git_status": subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        text=True,
    ).splitlines(),
    "dataset_yaml": str(data_yaml.resolve()),
    "dataset_yaml_sha256": sha256(data_yaml),
    "dataset_root": os.environ["DATA_ROOT"],
    "integrity_summary_sha256": sha256(integrity_summary),
    "epochs": 100,
    "imgsz": 640,
    "batch": 32,
    "nbs": 64,
    "workers": 8,
    "cache": False,
    "device": "0",
    "save_period": 5,
    "command": os.environ["TRAIN_COMMAND_TEXT"].strip(),
    "run_dir": os.environ["RUN_DIR"],
    "log": os.environ["INNER_LOG"],
    "status_log": os.environ["STATUS_FILE"],
    "free_vram_mib_at_start": int(os.environ["FREE_VRAM_MIB"]),
    "filesystem_available_kib_at_start": int(os.environ["AVAILABLE_KIB"]),
    "resume_supported": False,
}

text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
registry_path.write_text(text, encoding="utf-8")
current_path.write_text(text, encoding="utf-8")
PY
}

timestamp() {
  date '+%Y-%m-%d %H:%M:%S %z'
}

status() {
  printf '[%s] %s\n' "$(timestamp)" "$*" | tee -a "$STATUS_FILE"
}

printf '%s\n' "$$" > "$PID_FILE"
: > "$STATUS_FILE"
write_registry "starting"

status "B0 FULL100 STARTING"
status "Run: $RUN_NAME"
status "PID: $$"
status "Commit: $(git rev-parse HEAD)"
status "Output: $RUN_DIR"
status "Log: $INNER_LOG"
write_registry "running"

set +e
"${TRAIN_COMMAND[@]}" 2>&1 | tee "$INNER_LOG"
TRAIN_RC="${PIPESTATUS[0]}"
set -e

if [[ "$TRAIN_RC" -ne 0 ]]; then
  status "B0 FULL100 FAILED: exit_code=$TRAIN_RC"
  write_registry "failed" "$TRAIN_RC" "$(date --iso-8601=seconds)"
  exit "$TRAIN_RC"
fi

required_artifacts=(
  "$RUN_DIR/weights/last.pt"
  "$RUN_DIR/weights/best.pt"
  "$RUN_DIR/results.csv"
  "$RUN_DIR/args.yaml"
  "$RUN_DIR/full_training_runtime.json"
)

for artifact in "${required_artifacts[@]}"; do
  if [[ ! -s "$artifact" ]]; then
    status "B0 FULL100 FAILED: missing artifact $artifact"
    write_registry "failed" "30" "$(date --iso-8601=seconds)"
    exit 30
  fi
done

status "B0 FULL100 COMPLETED"
status "Final checkpoint: $RUN_DIR/weights/last.pt"
write_registry "completed" "0" "$(date --iso-8601=seconds)"
