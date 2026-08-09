#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_ROOT"

if [[ "$(git branch --show-current)" != "AnKy06" ]]; then
    echo "Dataset pipeline requires branch AnKy06" >&2
    exit 2
fi

PYTHON="$PROJECT_ROOT/.venv-fitlab02/bin/python"
if [[ ! -x "$PYTHON" ]]; then
    echo "Missing FITLAB-02 Python: $PYTHON" >&2
    exit 2
fi

RUN_ID="dataset_pipeline_$(date -u +%Y%m%d_%H%M%S)"
LOG_ROOT="$PROJECT_ROOT/outputs/datasets/FITLAB-02/$RUN_ID"
REGISTRY_ROOT="$PROJECT_ROOT/outputs/registry/FITLAB-02"
LOCK_FILE="$REGISTRY_ROOT/dataset_pipeline.lock"
STATUS_FILE="$REGISTRY_ROOT/dataset_pipeline_current.txt"
mkdir -p "$LOG_ROOT" "$REGISTRY_ROOT"

exec 9>"$LOCK_FILE"
if ! flock -n 9; then
    echo "Another dataset pipeline is already running" >&2
    exit 3
fi

write_status() {
    local state="$1"
    local step="$2"
    local temporary="$STATUS_FILE.tmp"
    {
        printf 'run_id=%s\n' "$RUN_ID"
        printf 'state=%s\n' "$state"
        printf 'step=%s\n' "$step"
        printf 'updated_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
        printf 'log_root=%s\n' "$LOG_ROOT"
        printf 'commit=%s\n' "$(git rev-parse HEAD)"
    } >"$temporary"
    mv "$temporary" "$STATUS_FILE"
}

run_step() {
    local step="$1"
    shift
    write_status "running" "$step"
    echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] START $step"
    "$@" >"$LOG_ROOT/$step.log" 2>&1
    echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] DONE  $step"
}

on_error() {
    local exit_code=$?
    write_status "failed" "${CURRENT_STEP:-unknown}"
    echo "Dataset pipeline failed with exit code $exit_code" >&2
    exit "$exit_code"
}
trap on_error ERR

write_status "waiting" "b0_training"
while pgrep -f '[t]rain_voc_full.py.*--baseline B0' >/dev/null; do
    echo "B0 training is active; dataset pipeline waits 60 seconds"
    sleep 60
done

CURRENT_STEP="01_voc_integrity"
run_step "$CURRENT_STEP" "$PYTHON" scripts/check_voc_yolo_integrity.py
CURRENT_STEP="02_voc_qa"
run_step "$CURRENT_STEP" "$PYTHON" scripts/qa_yolo_dataset.py \
    --dataset-id voc
CURRENT_STEP="03_inventory_after_voc"
run_step "$CURRENT_STEP" "$PYTHON" scripts/check_datasets.py

CURRENT_STEP="04_exdark_convert"
run_step "$CURRENT_STEP" "$PYTHON" scripts/convert_exdark_to_yolo.py --force
CURRENT_STEP="05_exdark_qa"
run_step "$CURRENT_STEP" "$PYTHON" scripts/qa_yolo_dataset.py \
    --dataset-id exdark

CURRENT_STEP="06_visdrone_convert"
run_step "$CURRENT_STEP" "$PYTHON" scripts/convert_visdrone_to_yolo.py --force
CURRENT_STEP="07_visdrone_qa"
run_step "$CURRENT_STEP" "$PYTHON" scripts/qa_yolo_dataset.py \
    --dataset-id visdrone2019

CURRENT_STEP="08_coco2017_convert"
run_step "$CURRENT_STEP" "$PYTHON" scripts/convert_coco2017_to_yolo.py \
    --workers 8 --force
CURRENT_STEP="09_coco2017_qa"
run_step "$CURRENT_STEP" "$PYTHON" scripts/qa_yolo_dataset.py \
    --dataset-id coco2017

CURRENT_STEP="10_bdd100k_download"
run_step "$CURRENT_STEP" bash scripts/download_bdd100k_mirror.sh \
    "$PROJECT_ROOT"
CURRENT_STEP="11_bdd100k_convert"
run_step "$CURRENT_STEP" "$PYTHON" scripts/convert_bdd100k_to_yolo.py \
    --workers 8 --force
CURRENT_STEP="12_bdd100k_qa"
run_step "$CURRENT_STEP" "$PYTHON" scripts/qa_yolo_dataset.py \
    --dataset-id bdd100k

CURRENT_STEP="13_voc_corruptions"
run_step "$CURRENT_STEP" "$PYTHON" scripts/build_voc_corruptions.py \
    --workers 4
CURRENT_STEP="14_voc_corruptions_qa"
run_step "$CURRENT_STEP" "$PYTHON" scripts/qa_voc_corruptions.py

CURRENT_STEP="15_final_inventory"
run_step "$CURRENT_STEP" "$PYTHON" scripts/check_datasets.py
write_status "complete" "$CURRENT_STEP"
echo "DATASET PIPELINE COMPLETE: $RUN_ID"
