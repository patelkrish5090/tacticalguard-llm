#!/usr/bin/env bash
#SBATCH -J tg_llm_h100
#SBATCH -p gpu_h100_4
#SBATCH --gres=gpu:1
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH -t 24:00:00
#SBATCH -o slurm.%j.out
#SBATCH -e slurm.%j.err

set -Eeuo pipefail

echo "Starting TacticalGuard-LLM H100 job"
echo "Host: $(hostname)"
echo "Node: ${SLURMD_NODENAME:-unknown}"
echo "Job ID: ${SLURM_JOB_ID:-manual}"
date

# Always run from the directory where sbatch was submitted.
PROJECT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "${PROJECT_DIR}"

VENV_DIR="${PROJECT_DIR}/venv_tacticalguard"
PYTHON="${VENV_DIR}/bin/python"
PIP="${VENV_DIR}/bin/pip"

if [ ! -x "${PYTHON}" ]; then
    echo "ERROR: venv Python not found or not executable: ${PYTHON}" >&2
    echo "Run this first on the login node: bash setup_hpc.sh" >&2
    echo "Project directory seen by job: ${PROJECT_DIR}" >&2
    ls -la "${PROJECT_DIR}" >&2
    exit 1
fi

# Do not depend on shell activation or module initialization. Put the venv first
# so subprocesses that call python/pip still resolve to this environment.
export VIRTUAL_ENV="${VENV_DIR}"
export PATH="${VENV_DIR}/bin:${PATH:-}"
export PYTHONNOUSERSITE=1
export TOKENIZERS_PARALLELISM=false
export HF_HOME="${PROJECT_DIR}/.hf_cache"
export TRANSFORMERS_CACHE="${HF_HOME}/transformers"
export SENTENCE_TRANSFORMERS_HOME="${HF_HOME}/sentence_transformers"

mkdir -p results data logs paper/tables results/figures "${HF_HOME}"

echo "Project directory: ${PROJECT_DIR}"
echo "Python: ${PYTHON}"
"${PYTHON}" --version
which python || true

if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi
else
    echo "WARNING: nvidia-smi not found in PATH"
fi

if [ -f ".env" ]; then
    echo "Loading environment variables from .env"
    set -a
    # shellcheck disable=SC1091
    source .env
    set +a
else
    echo "ERROR: .env file not found. Run setup_hpc.sh and add HF_TOKEN." >&2
    exit 1
fi

if [ -z "${HF_TOKEN:-}" ] || [ "${HF_TOKEN}" = "your_huggingface_token_here" ]; then
    echo "ERROR: HF_TOKEN is missing or still placeholder in .env" >&2
    exit 1
fi

echo "Verifying Python environment"
"${PYTHON}" - <<'PY'
import sys
print("executable:", sys.executable)
import numpy
print("numpy:", numpy.__version__)
import torch
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0))
import transformers
print("transformers:", transformers.__version__)
PY

echo "Running tests"
"${PYTHON}" -u -m pytest tests/ -v --tb=short

echo "Generating clean data and fitting anomaly filter"
"${PYTHON}" -u scripts/generate_clean_data.py --n_episodes 200 --n_steps 50

echo "Running experiments A-E with local LLaMA"
"${PYTHON}" -u run_all_experiments.py \
    --n_episodes 50 \
    --conditions A B C D E \
    --skip_openai \
    --agent_model local_llm \
    --n_steps 100

if [ -n "${OPENAI_API_KEY:-}" ] && [ "${OPENAI_API_KEY}" != "your_openai_api_key_here" ]; then
    echo "Running OpenAI condition F"
    "${PYTHON}" -u run_all_experiments.py --n_episodes 50 --conditions F
else
    echo "Skipping condition F because OPENAI_API_KEY is not set"
fi

echo "Generating analysis plots"
"${PYTHON}" -u notebooks/analysis.py

echo "Job complete"
date
