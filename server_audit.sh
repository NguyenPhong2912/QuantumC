#!/usr/bin/env bash

set -u

echo "============================================================"
echo "Q-VisionFrame - FITLab 01 Server Audit"
echo "Date: $(date)"
echo "============================================================"

echo
echo "===== USER / HOST ====="
whoami
hostname
hostnamectl 2>/dev/null || true

echo
echo "===== OPERATING SYSTEM ====="
cat /etc/os-release 2>/dev/null || true
uname -a

echo
echo "===== CPU ====="
lscpu

echo
echo "===== MEMORY ====="
free -h

echo
echo "===== STORAGE ====="
df -h /export/users/1165521/iDragonCloud/QuantumC
df -Th /export/users/1165521/iDragonCloud/QuantumC
du -sh /export/users/1165521/iDragonCloud/QuantumC 2>/dev/null || true

echo
echo "===== GPU / NVIDIA DRIVER ====="
command -v nvidia-smi || true
nvidia-smi || true

echo
echo "===== CUDA TOOLKIT ====="
command -v nvcc || true
nvcc --version 2>/dev/null || true

echo
echo "===== CUDA DIRECTORIES ====="
ls -ld /usr/local/cuda* 2>/dev/null || true

echo
echo "===== PYTHON ====="
command -v python3 || true
python3 --version 2>&1 || true
python3 -c "import sys; print(sys.executable); print(sys.version)" 2>/dev/null || true

echo
echo "===== CONDA / MAMBA ====="
command -v conda || true
conda --version 2>/dev/null || true
command -v mamba || true
mamba --version 2>/dev/null || true
command -v micromamba || true
micromamba --version 2>/dev/null || true

echo
echo "===== COMPILERS ====="
gcc --version 2>/dev/null | head -n 1 || true
g++ --version 2>/dev/null | head -n 1 || true
cmake --version 2>/dev/null | head -n 1 || true

echo
echo "===== SLURM ====="
command -v srun || true
command -v sbatch || true
sinfo 2>/dev/null || true

echo
echo "===== ENVIRONMENT VARIABLES ====="
echo "PATH=$PATH"
echo "LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-}"
echo "CUDA_HOME=${CUDA_HOME:-}"
echo "CONDA_PREFIX=${CONDA_PREFIX:-}"

echo
echo "===== INTERNET / PACKAGE ACCESS ====="
python3 -m pip --version 2>/dev/null || true
git --version 2>/dev/null || true

echo
echo "===== DIRECTORY PERMISSION TEST ====="
touch .qvisionframe_write_test
if [ -f .qvisionframe_write_test ]; then
    echo "WRITE_PERMISSION=OK"
    rm -f .qvisionframe_write_test
else
    echo "WRITE_PERMISSION=FAILED"
fi

echo
echo "============================================================"
echo "Audit completed."
echo "============================================================"
