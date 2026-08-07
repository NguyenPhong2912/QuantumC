#!/usr/bin/env bash
set -u

ROOT="/export/users/1165521/iDragonCloud/QuantumC"

show_server() {
    local host="$1"
    local label="$2"

    echo
    echo "================================================================================"
    echo "$label — $(date '+%Y-%m-%d %H:%M:%S %z')"
    echo "================================================================================"

    ssh -o ConnectTimeout=5 "$host" bash -s <<'REMOTE'
ROOT="/export/users/1165521/iDragonCloud/QuantumC"
HOST_SHORT="$(hostname -s)"

echo "Host: $(hostname)"
echo

echo "===== GPU SUMMARY ====="
nvidia-smi \
  --query-gpu=index,name,utilization.gpu,memory.used,memory.total,power.draw,temperature.gpu \
  --format=csv

echo
echo "===== GPU PROCESSES ====="
nvidia-smi \
  --query-compute-apps=pid,process_name,used_gpu_memory \
  --format=csv,noheader || true

echo
echo "===== Q-VISIONFRAME GPU MAIN PROCESS ====="

QVF_GPU_PID=""

while IFS= read -r pid; do
    [[ -n "$pid" ]] || continue

    cmd="$(
      ps -p "$pid" -o args= 2>/dev/null || true
    )"

    if [[ "$cmd" == *"scripts/train_voc_full.py"* ]]; then
        QVF_GPU_PID="$pid"

        ps -p "$pid" \
          -o pid,ppid,etime,%cpu,%mem,stat,args
    fi
done < <(
    nvidia-smi \
      --query-compute-apps=pid \
      --format=csv,noheader,nounits \
      2>/dev/null
)

if [[ -z "$QVF_GPU_PID" ]]; then
    echo "No Q-VisionFrame GPU main process detected"
fi

echo
echo "===== PIPELINE WRAPPER ====="

ps -eo pid,ppid,etime,%cpu,%mem,stat,args |
awk '
  /bash .*run_b7_pilot_then_full100_fitlab01\.sh/ &&
  $0 !~ /awk/ {
      print
      found=1
  }
  END {
      if (!found) {
          print "No B7 pipeline wrapper detected"
      }
  }
'

echo
echo "===== HOST-LOCAL PID STATUS ====="

case "$HOST_SHORT" in
    FITLAB-01*)
        pid_files=(
          "$ROOT/outputs/pids/b7_pilot_then_full100_fitlab01.pid"
        )
        ;;
    FITLAB-02*)
        pid_files=(
          "$ROOT/outputs/pids/train_voc_b8_resume_epoch6.pid"
          "$ROOT/outputs/pids/train_voc_b8_seed42_full100.pid"
        )
        ;;
    *)
        pid_files=()
        ;;
esac

if [[ "${#pid_files[@]}" -eq 0 ]]; then
    echo "No PID files configured for this host"
else
    for file in "${pid_files[@]}"; do
        if [[ ! -f "$file" ]]; then
            printf '%-55s %s\n' \
              "$(basename "$file")" \
              "MISSING"
            continue
        fi

        pid="$(cat "$file" 2>/dev/null || true)"

        if [[ -n "$pid" ]] &&
           kill -0 "$pid" 2>/dev/null; then
            state="RUNNING"
        else
            state="COMPLETED/STALE"
        fi

        printf '%-55s PID=%-10s %s\n' \
          "$(basename "$file")" \
          "$pid" \
          "$state"
    done
fi

echo
echo "===== RECENT TRAINING LOGS ====="

case "$HOST_SHORT" in
    FITLAB-01*)
        logs=(
          "$ROOT/outputs/logs/FITLAB-01/b7_pilot_then_full100.pipeline.log"
          "$ROOT/outputs/logs/FITLAB-01/train_voc_b7_seed42_full100_b32.inner.log"
        )
        ;;
    FITLAB-02*)
        logs=(
          "$ROOT/outputs/logs/FITLAB-02/train_voc_b8_seed42_full100.log"
        )
        ;;
    *)
        logs=()
        ;;
esac

found=0

for log in "${logs[@]}"; do
    if [[ -f "$log" ]]; then
        found=1
        echo
        echo "--- $log ---"
        tail -n 12 "$log"
    fi
done

if [[ "$found" -eq 0 ]]; then
    echo "No expected training log found"
fi
REMOTE
}

while true; do
    clear

    show_server fitlab-01 FITLAB-01
    show_server fitlab-02 FITLAB-02

    echo
    echo "Refresh every 30 seconds — Ctrl+C to exit"
    sleep 30
done
