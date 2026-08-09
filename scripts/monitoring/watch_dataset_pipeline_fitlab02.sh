#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
STATUS_FILE="$PROJECT_ROOT/outputs/registry/FITLAB-02/dataset_pipeline_current.txt"

if [[ ! -f "$STATUS_FILE" ]]; then
    echo "Dataset pipeline has not been started"
    exit 1
fi

cat "$STATUS_FILE"

LOG_ROOT="$(awk -F= '$1 == "log_root" {print substr($0, index($0, "=") + 1)}' "$STATUS_FILE")"
STEP="$(awk -F= '$1 == "step" {print substr($0, index($0, "=") + 1)}' "$STATUS_FILE")"

if [[ -n "$LOG_ROOT" && -n "$STEP" && -f "$LOG_ROOT/$STEP.log" ]]; then
    tail -n 25 "$LOG_ROOT/$STEP.log"
fi
