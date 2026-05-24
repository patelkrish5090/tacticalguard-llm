# TacticalGuard-LLM — Google Colab Runner
# ─────────────────────────────────────────────────────────────────────────────
# Run this file cell-by-cell in a Google Colab notebook.
# Required runtime: GPU (T4 free tier works for 4-bit LLaMA; A100 for full)
# Runtime → Change runtime type → GPU (T4 or A100)
#
# TOKENS YOU NEED:
#   1. HuggingFace Token  → https://huggingface.co/settings/tokens
#      (Also request LLaMA access: https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct)
#   2. OpenAI API Key     → https://platform.openai.com/api-keys
#      (ONLY needed for Condition F — cross-model test with GPT-4o-mini)
# ─────────────────────────────────────────────────────────────────────────────


# ╔══════════════════════════════════════════════════════════════════════╗
# ║  CELL 1 — Check GPU                                                 ║
# ╚══════════════════════════════════════════════════════════════════════╝
import subprocess
result = subprocess.run(["nvidia-smi"], capture_output=True, text=True)
print(result.stdout if result.returncode == 0 else "No GPU detected — switch to GPU runtime!")


# ╔══════════════════════════════════════════════════════════════════════╗
# ║  CELL 2 — Mount Google Drive (recommended: persists results)        ║
# ╚══════════════════════════════════════════════════════════════════════╝
from google.colab import drive
drive.mount('/content/drive')

# Project will live here on Drive:
PROJECT_DIR = "/content/drive/MyDrive/tacticalguard-llmv2"

import os
os.makedirs(PROJECT_DIR, exist_ok=True)
print(f"Project directory: {PROJECT_DIR}")


# ╔══════════════════════════════════════════════════════════════════════╗
# ║  CELL 3 — Clone / Upload the project                                ║
# ╚══════════════════════════════════════════════════════════════════════╝
# Option A: If you pushed to GitHub (recommended):
# !git clone https://github.com/YOUR_USERNAME/tacticalguard-llmv2.git /content/tacticalguard-llmv2

# Option B: Upload as zip from your local machine, then unzip:
# from google.colab import files
# uploaded = files.upload()   # select tacticalguard-llmv2.zip
# !unzip -q tacticalguard-llmv2.zip -d /content/

# Option C: Copy from Google Drive if you already uploaded the folder:
# !cp -r "/content/drive/MyDrive/tacticalguard-llmv2" /content/tacticalguard-llmv2

# After cloning/uploading, set the working directory:
import os
os.chdir("/content/tacticalguard-llmv2")  # adjust if needed
print("Working directory:", os.getcwd())


# ╔══════════════════════════════════════════════════════════════════════╗
# ║  CELL 4 — Install all dependencies                                  ║
# ╚══════════════════════════════════════════════════════════════════════╝
# Installs everything in requirements.txt + sentence-transformers extras
get_ipython().system('pip install -q torch>=2.2.0 transformers>=4.40.0 accelerate>=0.27.0')
get_ipython().system('pip install -q bitsandbytes>=0.43.0')
get_ipython().system('pip install -q sentence-transformers>=2.6.0')
get_ipython().system('pip install -q scikit-learn>=1.4.0 pandas>=2.2.0 numpy>=1.26.0')
get_ipython().system('pip install -q pyyaml>=6.0.1 tqdm>=4.66.0 openai>=1.23.0')
get_ipython().system('pip install -q matplotlib>=3.8.0 seaborn>=0.13.0 jsonlines>=4.0.0')
print("All dependencies installed.")


# ╔══════════════════════════════════════════════════════════════════════╗
# ║  CELL 5 — Set secrets / tokens                                      ║
# ╚══════════════════════════════════════════════════════════════════════╝
import os

# ── HuggingFace Token (REQUIRED for LLaMA 3.1-8B) ────────────────────
# Get yours at: https://huggingface.co/settings/tokens
# Also request model access at: https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct
#
# Recommended: use Colab Secrets (🔑 icon in left sidebar):
#   Key name:  HF_TOKEN
#   Value:     hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxx
#
# Then read it securely:
from google.colab import userdata
try:
    os.environ["HF_TOKEN"] = userdata.get("HF_TOKEN")
    print("HF_TOKEN loaded from Colab Secrets ✓")
except Exception:
    # Fallback: paste directly (less secure)
    os.environ["HF_TOKEN"] = "hf_PASTE_YOUR_TOKEN_HERE"
    print("HF_TOKEN set manually (consider using Colab Secrets instead)")

# ── OpenAI API Key (OPTIONAL — only for Condition F: GPT-4o-mini) ─────
# Get yours at: https://platform.openai.com/api-keys
try:
    os.environ["OPENAI_API_KEY"] = userdata.get("OPENAI_API_KEY")
    print("OPENAI_API_KEY loaded from Colab Secrets ✓")
except Exception:
    os.environ["OPENAI_API_KEY"] = ""  # Leave empty to skip Condition F
    print("OPENAI_API_KEY not set — Condition F (GPT-4o-mini) will be skipped")

# ── HuggingFace login (authenticates model download) ─────────────────
from huggingface_hub import login
login(token=os.environ["HF_TOKEN"], add_to_git_credential=False)
print("HuggingFace login successful ✓")


# ╔══════════════════════════════════════════════════════════════════════╗
# ║  CELL 6 — (Optional) Install CAGE 4 real environment               ║
# ╚══════════════════════════════════════════════════════════════════════╝
# Skip this if you want to use the MockCAGE4Wrapper (faster, no install needed)
# Uncomment to install the real CAGE 4:
#
# !git clone https://github.com/cage-challenge/cage-challenge-4.git
# !pip install -q -e cage-challenge-4/
# print("CAGE 4 installed ✓")
print("Skipping real CAGE 4 install — using MockCAGE4Wrapper (fast mode)")


# ╔══════════════════════════════════════════════════════════════════════╗
# ║  CELL 7 — Run tests to verify everything works                      ║
# ╚══════════════════════════════════════════════════════════════════════╝
get_ipython().system('python -m pytest tests/ -v --tb=short 2>&1')


# ╔══════════════════════════════════════════════════════════════════════╗
# ║  CELL 8 — Generate clean data + fit anomaly filter                 ║
# ║  (Required before running conditions B-F)                           ║
# ╚══════════════════════════════════════════════════════════════════════╝
# ~3-5 minutes: generates 200 clean episodes + fits IsolationForest + tunes threshold
get_ipython().system('python scripts/generate_clean_data.py --n_episodes 200 --n_steps 50')


# ╔══════════════════════════════════════════════════════════════════════╗
# ║  CELL 9 — Run baseline (no attack, no defense — MockLLM)           ║
# ╚══════════════════════════════════════════════════════════════════════╝
get_ipython().system('python run_baseline.py --n_episodes 10 --n_steps 50 --agent_model mock')


# ╔══════════════════════════════════════════════════════════════════════╗
# ║  CELL 10 — Run baseline with REAL LLaMA 3.1-8B-Instruct (4-bit)   ║
# ║  Requires GPU + HF_TOKEN set above                                  ║
# ╚══════════════════════════════════════════════════════════════════════╝
# NOTE: First run will download ~5GB model weights. Subsequent runs use cache.
# T4 (free Colab) can handle 4-bit LLaMA 3.1-8B (~6GB VRAM needed)
get_ipython().system('python run_baseline.py --n_episodes 10 --n_steps 30 --agent_model local_llm')


# ╔══════════════════════════════════════════════════════════════════════╗
# ║  CELL 11 — Quick smoke test: all conditions with MockLLM (fast)    ║
# ║  Use this to verify pipeline before running full LLaMA experiments  ║
# ╚══════════════════════════════════════════════════════════════════════╝
get_ipython().system('python run_all_experiments.py --n_episodes 10 --conditions A B C D E --skip_openai')


# ╔══════════════════════════════════════════════════════════════════════╗
# ║  CELL 12 — FULL PAPER EXPERIMENTS with real LLaMA (50 episodes)   ║
# ║  ⚠ Expected runtime: ~8-12 hours per condition on T4               ║
# ║  ⚠ Use A100 (Colab Pro) for faster runs (~2-3 hours per condition) ║
# ╚══════════════════════════════════════════════════════════════════════╝
# First, update configs to use local_llm instead of mock:
import yaml, glob

for cfg_path in glob.glob("configs/*.yaml"):
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    cfg["agent_model"] = "local_llm"
    cfg["n_steps"] = 100
    with open(cfg_path, "w") as f:
        yaml.dump(cfg, f)
    print(f"Updated {cfg_path}: agent_model=local_llm, n_steps=100")

# Run conditions A-E (skip F if no OpenAI key)
get_ipython().system('python run_all_experiments.py --n_episodes 50 --conditions A B C D E --skip_openai')


# ╔══════════════════════════════════════════════════════════════════════╗
# ║  CELL 13 — Condition F: Cross-model (GPT-4o-mini)                  ║
# ║  Requires OPENAI_API_KEY                                            ║
# ╚══════════════════════════════════════════════════════════════════════╝
if os.environ.get("OPENAI_API_KEY"):
    get_ipython().system('python run_all_experiments.py --n_episodes 50 --conditions F')
else:
    print("OPENAI_API_KEY not set — skipping Condition F")
    print("To run: set OPENAI_API_KEY in Colab Secrets and re-run this cell")


# ╔══════════════════════════════════════════════════════════════════════╗
# ║  CELL 14 — Analysis: load results + print table                    ║
# ╚══════════════════════════════════════════════════════════════════════╝
import json, os
import pandas as pd

results = {}
for cond in "ABCDEF":
    path = f"results/condition_{cond}_scorecard.json"
    if os.path.exists(path):
        with open(path) as f:
            results[cond] = json.load(f)

if results:
    df = pd.DataFrame(results).T
    cols = ["CAR", "MTTF", "CatchRate", "FPR", "latency_mean_ms", "DST", "AER"]
    print(df[[c for c in cols if c in df.columns]].to_string())
else:
    print("No results yet — run experiments first")


# ╔══════════════════════════════════════════════════════════════════════╗
# ║  CELL 15 — Generate plots (saves to results/figures/)              ║
# ╚══════════════════════════════════════════════════════════════════════╝
get_ipython().system('python notebooks/analysis.py')


# ╔══════════════════════════════════════════════════════════════════════╗
# ║  CELL 16 — Copy results to Google Drive (backup)                   ║
# ╚══════════════════════════════════════════════════════════════════════╝
import shutil, os

DRIVE_RESULTS = "/content/drive/MyDrive/tacticalguard-results"
os.makedirs(DRIVE_RESULTS, exist_ok=True)

for item in ["results/", "data/filter_fitted.pkl", "paper/tables/results_table.tex"]:
    if os.path.exists(item):
        dest = os.path.join(DRIVE_RESULTS, os.path.basename(item.rstrip("/")))
        if os.path.isdir(item):
            shutil.copytree(item, dest, dirs_exist_ok=True)
        else:
            shutil.copy2(item, dest)
        print(f"Copied {item} → {dest}")

print("All results backed up to Google Drive ✓")
