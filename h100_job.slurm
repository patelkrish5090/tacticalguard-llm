#!/bin/bash
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

# ==============================================================================

echo "🚀 Starting TacticalGuard-LLM Job on H100..."
echo "Node: $SLURMD_NODENAME"
echo "Job ID: $SLURM_JOB_ID"
date

# 1. Load the exact same Python 3.10 module used during setup
module purge
module load python-3.10.8-gcc-11.2.0-dlcmq7k

# 2. Activate Virtual Environment
source venv_tacticalguard/bin/activate

# 3. Safely load tokens from .env file
if [ -f ".env" ]; then
    echo "🔑 Loading environment variables from .env..."
    # This reads the .env file, ignores comments/blank lines, and exports them
    export $(grep -v '^#' .env | xargs)
else
    echo "❌ ERROR: .env file not found. Run setup_hpc.sh first!"
    exit 1
fi

# 4. Verify HF_TOKEN is loaded
if [ -z "$HF_TOKEN" ] || [ "$HF_TOKEN" == "your_huggingface_token_here" ]; then
    echo "❌ ERROR: HF_TOKEN is missing or invalid in .env file."
    exit 1
fi

echo "✅ Environment ready. Running Pytest..."

# Create necessary directories
mkdir -p results data logs paper/tables results/figures

# Step A: Run tests to verify setup
$PWD/venv_tacticalguard/bin/python -m pytest tests/ -v --tb=short

# Step B: Generate clean data and fit anomaly filter
echo "📊 Generating clean data and fitting filter..."
$PWD/venv_tacticalguard/bin/python scripts/generate_clean_data.py --n_episodes 200 --n_steps 50

# Step C: Run the experiments
# Depending on how long you want to run, adjust --n_episodes (10 for test, 50 for paper)
echo "🔬 Running Full Experiments (Conditions A-E)..."
$PWD/venv_tacticalguard/bin/python run_all_experiments.py --n_episodes 50 --conditions A B C D E --skip_openai --agent_model local_llm

# Step D: (Optional) Condition F with OpenAI
if [ -n "$OPENAI_API_KEY" ] && [ "$OPENAI_API_KEY" != "your_openai_api_key_here" ]; then
    echo "🔬 Running OpenAI Cross-model Experiment (Condition F)..."
    $PWD/venv_tacticalguard/bin/python run_all_experiments.py --n_episodes 50 --conditions F
fi

# Step E: Generate Plots
echo "📈 Generating analysis plots..."
$PWD/venv_tacticalguard/bin/python notebooks/analysis.py

echo "🎉 All Done!"
date
