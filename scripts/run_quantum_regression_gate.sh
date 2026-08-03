#!/usr/bin/env bash
set -euo pipefail

ROOT="/export/users/1165521/iDragonCloud/QuantumC"
cd "$ROOT"

: "${QVF_SERVER:=FITLAB-02}"
: "${QVF_LOG_DIR:=$ROOT/outputs/logs/$QVF_SERVER}"

mkdir -p "$QVF_LOG_DIR"

echo "===== Q-VISIONFRAME QUANTUM REGRESSION GATE ====="
echo "Server : $QVF_SERVER"
echo "Root   : $ROOT"
echo

echo "===== 1. PYTHON COMPILE ====="

python -m py_compile \
  src/qvisionframe/pqfg.py \
  src/qvisionframe/quantum_detection_model.py \
  src/qvisionframe/quantum_checkpoint.py \
  src/qvisionframe/quantum_detection_trainer.py \
  src/qvisionframe/scientific_state_hash.py \
  scripts/smoke_train_voc_quantum.py \
  scripts/smoke_resume_voc_quantum.py \
  scripts/audit_quantum_production_matrix.py \
  tests/test_scientific_checkpoint_integrity.py

echo "PYTHON COMPILE: PASSED"
echo

echo "===== 2. SCIENTIFIC CHECKPOINT INTEGRITY ====="

python tests/test_scientific_checkpoint_integrity.py

echo

echo "===== 3. QUANTUM PRODUCTION MATRIX ====="

python scripts/audit_quantum_production_matrix.py

echo

echo "===== 4. REQUIRED ARTIFACTS ====="

python - <<'PY'
from pathlib import Path

root = Path(
    "/export/users/1165521/iDragonCloud/QuantumC"
)

required = [
    root / (
        "outputs/inventory/"
        "quantum_production_matrix.json"
    ),
    root / (
        "outputs/inventory/"
        "scientific_integrity/"
        "b8_signed_checkpoint.pt"
    ),
    root / (
        "outputs/inventory/"
        "scientific_integrity/"
        "b8_tampered_checkpoint.pt"
    ),
]

for path in required:
    print(path, ":", path.exists())
    assert path.exists(), path

print("REQUIRED ARTIFACTS: PASSED")
PY

echo
echo "Q-VISIONFRAME QUANTUM REGRESSION GATE: PASSED"
