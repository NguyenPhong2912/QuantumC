#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(
  cd "$(dirname "${BASH_SOURCE[0]}")/../.."
  pwd
)"

MODE="${1:---watch}"
INTERVAL="${QVF_MONITOR_INTERVAL:-30}"
PYTHON="$ROOT/.venv-fitlab02/bin/python"
PID_FILE="$ROOT/outputs/pids/train_voc_b0_full100.pid"
REGISTRY_FILE="$ROOT/outputs/registry/FITLAB-02/b0_current.json"

if [[ "$MODE" != "--watch" && "$MODE" != "--once" ]]; then
  echo "Usage: $0 [--watch|--once]" >&2
  exit 2
fi

if [[ ! -x "$PYTHON" ]]; then
  echo "ERROR: Python environment not found: $PYTHON" >&2
  exit 10
fi

show_status() {
  if [[ "$MODE" == "--watch" ]]; then
    clear
  fi

  echo "===== B0 FITLAB-02 MONITOR ====="
  echo "Time: $(date '+%Y-%m-%d %H:%M:%S %z')"
  echo

  nvidia-smi \
    --query-gpu=index,name,utilization.gpu,memory.used,memory.total,power.draw,temperature.gpu \
    --format=csv

  echo
  echo "===== PID ====="

  if [[ -f "$PID_FILE" ]]; then
    pid="$(cat "$PID_FILE")"

    if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
      ps -p "$pid" -o pid,ppid,etime,%cpu,%mem,stat,args
    else
      echo "PID $pid is completed or stale"
    fi
  else
    echo "PID file not found"
  fi

  echo
  echo "===== REGISTRY ====="

  if [[ -f "$REGISTRY_FILE" ]]; then
    "$PYTHON" - "$REGISTRY_FILE" <<'PY'
from pathlib import Path
import json
import sys

payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))

for key in (
    "run_name",
    "status",
    "start_time",
    "end_time",
    "exit_code",
    "git_commit",
    "run_dir",
    "log",
):
    print(f"{key:12}: {payload.get(key)}")
PY

    current_log="$($PYTHON - "$REGISTRY_FILE" <<'PY'
from pathlib import Path
import json
import sys

payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
print(payload.get("log") or "")
PY
)"

    if [[ -n "$current_log" && -f "$current_log" ]]; then
      echo
      echo "===== RECENT TRAINING LOG ====="
      tail -n 20 "$current_log"
    fi
  else
    echo "Registry not found"
  fi
}

while true; do
  show_status

  if [[ "$MODE" == "--once" ]]; then
    exit 0
  fi

  sleep "$INTERVAL"
done
