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

SIF="${PROJECT_DIR}/tacticalguard.sif"

# Verify the container image exists
if [ ! -f "${SIF}" ]; then
    echo "ERROR: Container image not found: ${SIF}" >&2
    echo "Run this first on the login node: bash build_container.sh" >&2
    exit 1
fi

# Cache directories (inside project so they persist across jobs)
export HF_HOME="${PROJECT_DIR}/.hf_cache"
export TRANSFORMERS_CACHE="${HF_HOME}/transformers"
export SENTENCE_TRANSFORMERS_HOME="${HF_HOME}/sentence_transformers"

mkdir -p results data logs paper/tables results/figures "${HF_HOME}"

# Load secrets from .env
if [ -f ".env" ]; then
    echo "Loading environment variables from .env"
    set -a
    source .env
    set +a
else
    echo "ERROR: .env file not found. Add your HF_TOKEN to a .env file." >&2
    exit 1
fi

if [ -z "${HF_TOKEN:-}" ] || [ "${HF_TOKEN}" = "your_huggingface_token_here" ]; then
    echo "ERROR: HF_TOKEN is missing or still placeholder in .env" >&2
    exit 1
fi

# Helper: run a command inside the container with GPU passthrough
# --nv            : passes H100 GPU into the container
# --bind          : mounts project dir to /workspace inside container
# --env           : passes secrets into the container environment
APPTAINER_RUN="apptainer exec \
    --nv \
    --bind ${PROJECT_DIR}:/workspace \
    --env HF_TOKEN=${HF_TOKEN} \
    --env OPENAI_API_KEY=${OPENAI_API_KEY:-} \
    --env HF_HOME=/workspace/.hf_cache \
    --env TRANSFORMERS_CACHE=/workspace/.hf_cache/transformers \
    --env TOKENIZERS_PARALLELISM=false \
    --env PYTHONUNBUFFERED=1 \
    --pwd /workspace \
    ${SIF}"

# Verify container + GPU
echo "Verifying container environment"
${APPTAINER_RUN} python -c "
import sys, numpy, torch, transformers
print('python:', sys.version)
print('numpy:', numpy.__version__)
print('torch:', torch.__version__)
print('cuda available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('GPU:', torch.cuda.get_device_name(0))
    print('VRAM:', round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1), 'GB')
print('transformers:', transformers.__version__)
"

# Step A: Run tests
echo "Running tests"
${APPTAINER_RUN} python -u -m pytest tests/ -v --tb=short

# Step B: Generate clean data and fit anomaly filter
echo "Generating clean data and fitting anomaly filter"
${APPTAINER_RUN} python -u scripts/generate_clean_data.py --n_episodes 200 --n_steps 50

# Step C: Run experiments A-E with local LLaMA
echo "Running experiments A-E with local LLaMA"
${APPTAINER_RUN} python -u run_all_experiments.py \
    --n_episodes 50 \
    --conditions A B C D E \
    --skip_openai \
    --agent_model local_llm \
    --n_steps 100

# Step D: (Optional) Condition F with OpenAI
if [ -n "${OPENAI_API_KEY:-}" ] && [ "${OPENAI_API_KEY}" != "your_openai_api_key_here" ]; then
    echo "Running OpenAI condition F"
    ${APPTAINER_RUN} python -u run_all_experiments.py --n_episodes 50 --conditions F
else
    echo "Skipping condition F (OPENAI_API_KEY not set)"
fi

# Step E: Generate analysis plots
echo "Generating analysis plots"
${APPTAINER_RUN} python -u notebooks/analysis.py

echo "Job complete"
date
