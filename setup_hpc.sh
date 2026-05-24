#!/usr/bin/env bash
# ==============================================================================
# setup_hpc.sh - TacticalGuard-LLM environment setup for HPC
# ==============================================================================
# Run once from the project directory on the login node:
#
#   bash setup_hpc.sh
#
# This script does not require environment modules. If your cluster provides
# modules, it will use them when available; otherwise it falls back to python3.
# ==============================================================================

set -Eeuo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${PROJECT_DIR}/venv_tacticalguard"

cd "${PROJECT_DIR}"

echo "Starting TacticalGuard-LLM HPC setup"
echo "Project directory: ${PROJECT_DIR}"

load_python_module_if_available() {
    if [ -f /etc/profile.d/modules.sh ]; then
        # shellcheck disable=SC1091
        source /etc/profile.d/modules.sh || true
    elif [ -f /usr/share/Modules/init/bash ]; then
        # shellcheck disable=SC1091
        source /usr/share/Modules/init/bash || true
    elif [ -f /usr/share/lmod/lmod/init/bash ]; then
        # shellcheck disable=SC1091
        source /usr/share/lmod/lmod/init/bash || true
    fi

    if command -v module >/dev/null 2>&1; then
        echo "Environment modules detected; trying Python module."
        module purge || true
        module load python-3.10.8-gcc-11.2.0-dlcmq7k || module load python/3.10 || module load python || true
    else
        echo "Environment modules not available; using system Python."
    fi
}

find_python() {
    if command -v python3 >/dev/null 2>&1; then
        command -v python3
    elif command -v python >/dev/null 2>&1; then
        command -v python
    else
        echo "ERROR: no python3/python command found. Load a Python module manually, then rerun." >&2
        exit 1
    fi
}

load_python_module_if_available
BASE_PYTHON="$(find_python)"

echo "Base Python: ${BASE_PYTHON}"
"${BASE_PYTHON}" --version

if [ ! -d "${VENV_DIR}" ]; then
    echo "Creating virtual environment: ${VENV_DIR}"
    "${BASE_PYTHON}" -m venv "${VENV_DIR}"
else
    echo "Virtual environment already exists: ${VENV_DIR}"
fi

VENV_PYTHON="${VENV_DIR}/bin/python"
VENV_PIP="${VENV_DIR}/bin/pip"

if [ ! -x "${VENV_PYTHON}" ]; then
    echo "ERROR: venv Python is missing or not executable: ${VENV_PYTHON}" >&2
    exit 1
fi

echo "Venv Python: ${VENV_PYTHON}"
"${VENV_PYTHON}" --version

echo "Upgrading pip, setuptools, and wheel"
"${VENV_PYTHON}" -m pip install --upgrade pip setuptools wheel

echo "Installing core numeric stack first"
"${VENV_PYTHON}" -m pip install -U "numpy>=1.26.0" "pandas>=2.2.0" "scipy" "scikit-learn>=1.4.0"

echo "Installing PyTorch for CUDA 12.1/H100"
"${VENV_PYTHON}" -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

echo "Installing project dependencies"
"${VENV_PYTHON}" -m pip install -r requirements.txt
"${VENV_PYTHON}" -m pip install -U "typing-extensions>=4.14.0" "numpy>=1.26.0" pytest

echo "Checking critical imports"
"${VENV_PYTHON}" - <<'PY'
import sys
print("python:", sys.executable)
import numpy
print("numpy:", numpy.__version__)
import torch
print("torch:", torch.__version__, "cuda:", torch.cuda.is_available())
import transformers
print("transformers:", transformers.__version__)
import sentence_transformers
print("sentence-transformers: ok")
PY

if [ ! -f ".env" ]; then
    cat > .env <<'EOF'
HF_TOKEN=your_huggingface_token_here
OPENAI_API_KEY=your_openai_api_key_here
EOF
    chmod 600 .env
    echo "Created .env. Edit it and add your Hugging Face token before real LLaMA jobs."
else
    echo ".env already exists."
fi

mkdir -p results data logs paper/tables results/figures

echo ""
echo "Setup complete."
echo "Submit with: sbatch h100_job.slurm"
echo "Manual sanity check:"
echo "  ${VENV_PYTHON} -m pytest tests/ -q"
