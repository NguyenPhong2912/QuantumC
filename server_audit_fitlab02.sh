#!/usr/bin/env bash

set -u

REPORT="server_audit_fitlab02.txt"

{
echo "============================================================"
echo "Q-VisionFrame - FITLab 02 Server Audit"
echo "Date: $(date)"
echo "============================================================"

echo
echo "===== USER / HOST ====="
whoami
hostname
hostname -f 2>/dev/null || true
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
grep -E 'MemTotal|MemAvailable|SwapTotal|SwapFree' /proc/meminfo

echo
echo "===== STORAGE ====="
df -h
echo
df -Th
echo
echo "Project directory:"
du -sh /export/users/1165521/iDragonCloud/QuantumC 2>/dev/null || true
echo
echo "Filesystem mount:"
findmnt -T /export/users/1165521/iDragonCloud/QuantumC 2>/dev/null || true

echo
echo "===== GPU / NVIDIA DRIVER ====="
command -v nvidia-smi || true
nvidia-smi || true

echo
echo "===== GPU DETAILED QUERY ====="
nvidia-smi \
    --query-gpu=index,name,uuid,driver_version,memory.total,memory.used,memory.free,temperature.gpu,power.draw,power.limit,utilization.gpu,compute_cap \
    --format=csv 2>/dev/null || true

echo
echo "===== GPU PROCESSES ====="
nvidia-smi pmon -c 1 2>/dev/null || true

echo
echo "===== CUDA TOOLKIT ====="
command -v nvcc || true
nvcc --version 2>/dev/null || true
which nvcc 2>/dev/null || true
readlink -f "$(command -v nvcc)" 2>/dev/null || true

echo
echo "===== CUDA INSTALLATIONS ====="
ls -ld /usr/local/cuda* 2>/dev/null || true
find /usr/local -maxdepth 2 -type f -name nvcc 2>/dev/null || true
ldconfig -p 2>/dev/null | grep -Ei 'cuda|cudnn|cublas|cusparse|cuquantum|custatevec' || true

echo
echo "===== CUDNN ====="
find /usr /usr/local -iname 'libcudnn*' 2>/dev/null | head -n 30 || true

echo
echo "===== CUQUANTUM / CUSTATEVEC ====="
find /usr /usr/local \
    \( -iname 'libcustatevec*' -o -iname 'libcutensornet*' -o -iname '*cuquantum*' \) \
    2>/dev/null | head -n 50 || true

echo
echo "===== PYTHON ====="
command -v python3 || true
python3 --version 2>&1 || true
python3 - <<'PY' 2>/dev/null || true
import platform
import site
import sys

print("Executable :", sys.executable)
print("Version    :", sys.version.replace("\n", " "))
print("Platform   :", platform.platform())
print("Prefix     :", sys.prefix)
print("Site dirs  :", site.getsitepackages())
PY

echo
echo "===== PIP ====="
python3 -m pip --version 2>/dev/null || true
python3 -m pip config list 2>/dev/null || true

echo
echo "===== CONDA / MAMBA ====="
command -v conda || true
conda --version 2>/dev/null || true
command -v mamba || true
mamba --version 2>/dev/null || true
command -v micromamba || true
micromamba --version 2>/dev/null || true

echo
echo "===== COMPILERS / BUILD TOOLS ====="
gcc --version 2>/dev/null | head -n 1 || true
g++ --version 2>/dev/null | head -n 1 || true
cmake --version 2>/dev/null | head -n 1 || true
make --version 2>/dev/null | head -n 1 || true
ninja --version 2>/dev/null || true

echo
echo "===== GIT ====="
git --version 2>/dev/null || true

echo
echo "===== CONTAINER TOOLS ====="
command -v docker || true
docker --version 2>/dev/null || true
command -v apptainer || true
apptainer --version 2>/dev/null || true
command -v singularity || true
singularity --version 2>/dev/null || true

echo
echo "===== SCHEDULER ====="
command -v srun || true
command -v sbatch || true
command -v qsub || true
sinfo 2>/dev/null || true

echo
echo "===== NETWORK ====="
ip -brief address 2>/dev/null || true
ip route 2>/dev/null || true

echo
echo "===== ENVIRONMENT VARIABLES ====="
echo "PATH=$PATH"
echo "LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-}"
echo "CUDA_HOME=${CUDA_HOME:-}"
echo "CUDA_PATH=${CUDA_PATH:-}"
echo "CONDA_PREFIX=${CONDA_PREFIX:-}"
echo "VIRTUAL_ENV=${VIRTUAL_ENV:-}"

echo
echo "===== PROCESS AND RESOURCE LIMITS ====="
ulimit -a

echo
echo "===== WRITE PERMISSION TEST ====="
TEST_FILE=".qvisionframe_fitlab02_write_test_$$"

if touch "$TEST_FILE" 2>/dev/null; then
    echo "WRITE_PERMISSION=OK"
    rm -f "$TEST_FILE"
else
    echo "WRITE_PERMISSION=FAILED"
fi

echo
echo "===== EXISTING PROJECT ENVIRONMENT ====="
if [ -d ".venv" ]; then
    echo ".venv exists"
    ls -ld .venv
    .venv/bin/python --version 2>/dev/null || true

    .venv/bin/python - <<'PY' 2>/dev/null || true
packages = [
    "torch",
    "torchvision",
    "pennylane",
    "pennylane_lightning",
    "numpy",
    "scipy",
    "SALib",
]

for package in packages:
    try:
        module = __import__(package)
        print(f"{package:24s}: {getattr(module, '__version__', 'installed')}")
    except Exception as exc:
        print(f"{package:24s}: NOT AVAILABLE ({type(exc).__name__})")
PY
else
    echo ".venv does not exist"
fi

echo
echo "============================================================"
echo "FITLab 02 audit completed."
echo "============================================================"
} 2>&1 | tee "$REPORT"

echo
echo "Saved report:"
ls -lh "$REPORT"
