#!/bin/bash
#SBATCH --job-name=tg_llm_h100
#SBATCH --output=logs/slurm-%j.out
#SBATCH --error=logs/slurm-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G                 # Requesting 64GB RAM (Node has 1TB)
#SBATCH --time=24:00:00           # Max time for full paper runs
#
# ===== TARGETING H100 NODES =====
# Based on cluster specs: gpu[5-6] have 4x H100 SXM5 80GB
# If your cluster requires a partition name, add it here (e.g., #SBATCH --partition=gpu)
# Requesting 1 H100 GPU:
#SBATCH --nodelist=gpu5,gpu6      # Restrict to the H100 nodes
#SBATCH --gres=gpu:1              # Request 1 GPU

# ==============================================================================

echo "🚀 Starting TacticalGuard-LLM Job on H100..."
echo "Node: $SLURMD_NODENAME"
echo "Job ID: $SLURM_JOB_ID"
date

# 1. Load CUDA module if required by your cluster
# module load cuda/12.1  # Uncomment and adjust if your cluster uses modules

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
python -m pytest tests/ -v --tb=short

# Step B: Generate clean data and fit anomaly filter
echo "📊 Generating clean data and fitting filter..."
python scripts/generate_clean_data.py --n_episodes 200 --n_steps 50

# Step C: Run the experiments
# Depending on how long you want to run, adjust --n_episodes (10 for test, 50 for paper)
echo "🔬 Running Full Experiments (Conditions A-E)..."
python run_all_experiments.py --n_episodes 50 --conditions A B C D E --skip_openai

# Step D: (Optional) Condition F with OpenAI
if [ -n "$OPENAI_API_KEY" ] && [ "$OPENAI_API_KEY" != "your_openai_api_key_here" ]; then
    echo "🔬 Running OpenAI Cross-model Experiment (Condition F)..."
    python run_all_experiments.py --n_episodes 50 --conditions F
fi

# Step E: Generate Plots
echo "📈 Generating analysis plots..."
python notebooks/analysis.py

echo "🎉 All Done!"
date
