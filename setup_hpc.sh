#!/bin/bash
# ==============================================================================
# setup_hpc.sh - TacticalGuard-LLM Environment Setup for HPC
# ==============================================================================
# Run this script ONCE on the login node to set up your virtual environment
# and install all required dependencies safely.
#
# Usage: bash setup_hpc.sh
# ==============================================================================

echo "🚀 Starting TacticalGuard-LLM HPC setup..."

# 0. Load the required Python 3.10 module to guarantee environment consistency
module purge
module load python-3.10.8-gcc-11.2.0-dlcmq7k

# 1. Create and activate virtual environment
VENV_DIR="venv_tacticalguard"
if [ ! -d "$VENV_DIR" ]; then
    echo "📦 Creating virtual environment: $VENV_DIR..."
    python3 -m venv $VENV_DIR
else
    echo "✅ Virtual environment $VENV_DIR already exists."
fi

echo "🔄 Activating virtual environment..."
source $VENV_DIR/bin/activate

# 2. Upgrade pip and core tools
echo "⬆️ Upgrading pip..."
pip install --upgrade pip setuptools wheel

# 3. Fix typing-extensions (critical for pytest/CAGE4 compatibility)
echo "🔧 Installing typing-extensions..."
pip install -U "typing-extensions>=4.14.0"

# 4. Install PyTorch (optimized for H100 - CUDA 12.1 is generally recommended for H100)
# Note: Adjust the CUDA version if your cluster uses a different default CUDA module.
echo "🔥 Installing PyTorch (CUDA 12.1)..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 5. Install LLM and project dependencies
echo "🧠 Installing HuggingFace and LLM dependencies..."
pip install "transformers>=4.40.0" "accelerate>=0.27.0" "bitsandbytes>=0.43.0"
pip install "sentence-transformers>=2.6.0" "scikit-learn>=1.4.0"
pip install "pyyaml>=6.0.1" "tqdm>=4.66.0" "openai>=1.23.0"
pip install "matplotlib>=3.8.0" "seaborn>=0.13.0" "jsonlines>=4.0.0" "pandas>=2.2.0"

# 6. Install testing tools
echo "🧪 Installing testing dependencies..."
pip install pytest

# 7. Install CAGE 4 Simulator
if [ ! -d "cage-challenge-4" ]; then
    echo "🎮 Cloning and installing CAGE 4 simulator..."
    git clone https://github.com/cage-challenge/cage-challenge-4.git
    pip install -e cage-challenge-4/
    
    # Re-pin typing-extensions just in case CAGE downgraded it
    pip install -U "typing-extensions>=4.14.0"
else
    echo "✅ CAGE 4 already installed."
fi

# 8. Setup .env file for secrets if it doesn't exist
if [ ! -f ".env" ]; then
    echo "🔑 Creating blank .env file for tokens..."
    echo "HF_TOKEN=your_huggingface_token_here" > .env
    echo "OPENAI_API_KEY=your_openai_api_key_here" >> .env
    echo "⚠️  IMPORTANT: Please edit the .env file and add your HF_TOKEN before running jobs!"
else
    echo "✅ .env file already exists."
fi

echo ""
echo "🎉 Setup Complete!"
echo "Next steps:"
echo "1. Edit .env and paste your HuggingFace token (and optionally OpenAI key)."
echo "2. Submit your job to the H100 nodes using: sbatch h100_job.slurm"
