#!/usr/bin/env bash


PROJECT_ROOT="/export/users/1165521/iDragonCloud/QuantumC"
VENV_PATH="${PROJECT_ROOT}/.venv-fitlab02"

cd "$PROJECT_ROOT"
source "${VENV_PATH}/bin/activate"

HOST_SHORT="$(hostname | cut -d. -f1)"

case "$HOST_SHORT" in
    FITLAB-01|fitlab-01)
        export QVF_SERVER="FITLAB-01"
        export QVF_COMPUTE_MODE="cuda"
        ;;
    FITLAB-02|fitlab-02)
        export QVF_SERVER="FITLAB-02"
        export QVF_COMPUTE_MODE="cuda"
        ;;
    FITLAB-03|fitlab-03)
        export QVF_SERVER="FITLAB-03"
        export QVF_COMPUTE_MODE="cpu"
        ;;
    *)
        echo "Unsupported hostname: $(hostname)" >&2
        return 1 2>/dev/null || exit 1
        ;;
esac

export QVF_ROOT="$PROJECT_ROOT"
export QVF_OUTPUT_ROOT="${PROJECT_ROOT}/outputs/${QVF_SERVER}"
export QVF_RUNS_DIR="${QVF_OUTPUT_ROOT}/runs"
export QVF_CHECKPOINT_DIR="${QVF_OUTPUT_ROOT}/checkpoints"
export QVF_LOG_DIR="${QVF_OUTPUT_ROOT}/logs"
export QVF_METRICS_DIR="${QVF_OUTPUT_ROOT}/metrics"
export QVF_PROFILE_DIR="${QVF_OUTPUT_ROOT}/profiling"

mkdir -p \
    "$QVF_RUNS_DIR" \
    "$QVF_CHECKPOINT_DIR" \
    "$QVF_LOG_DIR" \
    "$QVF_METRICS_DIR" \
    "$QVF_PROFILE_DIR"

export PYTHONPATH="${QVF_ROOT}:${PYTHONPATH:-}"

echo "===== Q-VISIONFRAME SESSION ====="
echo "Server       : $QVF_SERVER"
echo "Hostname     : $(hostname)"
echo "Project root : $QVF_ROOT"
echo "Python       : $(which python)"
echo "Output root  : $QVF_OUTPUT_ROOT"
