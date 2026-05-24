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

import os
os.makedirs("/content/drive/MyDrive/tacticalguard-results", exist_ok=True)
print("Google Drive mounted ✓")


# ╔══════════════════════════════════════════════════════════════════════╗
# ║  CELL 3 — Clone / Upload the project                                ║
# ╚══════════════════════════════════════════════════════════════════════╝
# Option A: GitHub (recommended — push your project first)
# !git clone https://github.com/YOUR_USERNAME/tacticalguard-llmv2.git /content/tacticalguard-llmv2

# Option B: Upload zip
# from google.colab import files
# uploaded = files.upload()   # select tacticalguard-llmv2.zip
# !unzip -q tacticalguard-llmv2.zip -d /content/

# Option C: Already on Drive
# !cp -r "/content/drive/MyDrive/tacticalguard-llmv2" /content/tacticalguard-llmv2

import os, sys
os.chdir("/content/tacticalguard-llmv2")   # ← adjust if path differs
sys.path.insert(0, "/content/tacticalguard-llmv2")
print("Working directory:", os.getcwd())


# ╔══════════════════════════════════════════════════════════════════════╗
# ║  CELL 4 — Install dependencies (Colab-safe, no downgrades)         ║
# ║                                                                      ║
# ║  KEY INSIGHT: Colab already ships torch 2.10+cu128, numpy 2.x.     ║
# ║  Do NOT reinstall them — just add what's missing and upgrade        ║
# ║  typing-extensions (fixes the typeguard/pytest crash).             ║
# ╚══════════════════════════════════════════════════════════════════════╝

# Step 1: Fix typing-extensions FIRST — this is what breaks pytest
# typeguard 4.5.1 needs NoExtraItems from typing_extensions>=4.14.0
get_ipython().system('pip install -q -U "typing-extensions>=4.14.0"')

# Step 2: transformers stack (do NOT pin torch — Colab has 2.10 already)
get_ipython().system('pip install -q "transformers>=4.40.0" "accelerate>=0.27.0"')

# Step 3: bitsandbytes (0.49.2 is in Colab; needs torch>=2.3 which is satisfied)
get_ipython().system('pip install -q "bitsandbytes>=0.43.0"')

# Step 4: Sentence embeddings + anomaly filter
get_ipython().system('pip install -q "sentence-transformers>=2.6.0"')
get_ipython().system('pip install -q "scikit-learn>=1.4.0"')

# Step 5: Utilities (no numpy pin — Colab has 2.x which is fine)
get_ipython().system('pip install -q "pyyaml>=6.0.1" "tqdm>=4.66.0" "openai>=1.23.0"')
get_ipython().system('pip install -q "matplotlib>=3.8.0" "seaborn>=0.13.0" "jsonlines>=4.0.0" "pandas>=2.2.0"')

print("\nAll dependencies installed.")
print("Verifying critical imports...")

import_checks = [
    "import torch; print(f'  torch {torch.__version__} ✓')",
    "import transformers; print(f'  transformers {transformers.__version__} ✓')",
    "import sentence_transformers; print(f'  sentence-transformers ✓')",
    "import sklearn; print(f'  scikit-learn {sklearn.__version__} ✓')",
    "import numpy; print(f'  numpy {numpy.__version__} ✓')",
    "import typing_extensions; print(f'  typing-extensions {typing_extensions.__version__} ✓')",
]
import subprocess
for check in import_checks:
    r = subprocess.run(["python", "-c", check], capture_output=True, text=True)
    print(r.stdout.strip() if r.returncode == 0 else f"  ✗ {r.stderr.strip()[:80]}")


# ╔══════════════════════════════════════════════════════════════════════╗
# ║  CELL 5 — Install CAGE 4 real environment                          ║
# ║  The dependency conflicts shown are Colab-level WARNING only —     ║
# ║  they do NOT affect TacticalGuard-LLM code.                        ║
# ╚══════════════════════════════════════════════════════════════════════╝
get_ipython().system('git clone -q https://github.com/cage-challenge/cage-challenge-4.git')
get_ipython().system('pip install -q -e cage-challenge-4/')

# CRITICAL: CAGE 4 may downgrade typing-extensions again — re-pin it
get_ipython().system('pip install -q -U "typing-extensions>=4.14.0"')
print("CAGE 4 installed ✓  |  typing-extensions re-pinned ✓")


# ╔══════════════════════════════════════════════════════════════════════╗
# ║  CELL 6 — Set tokens securely via Colab Secrets                    ║
# ╚══════════════════════════════════════════════════════════════════════╝
# Add secrets via the 🔑 icon in the left sidebar:
#   Key: HF_TOKEN        Value: hf_xxxxxxxxxxxxxxxxxxxx
#   Key: OPENAI_API_KEY  Value: sk-proj-xxxxxxxxxxxxxxx  (optional)

import os
from google.colab import userdata

try:
    os.environ["HF_TOKEN"] = userdata.get("HF_TOKEN")
    print("HF_TOKEN loaded from Colab Secrets ✓")
except Exception:
    os.environ["HF_TOKEN"] = "hf_PASTE_YOUR_TOKEN_HERE"
    print("⚠  HF_TOKEN not in Secrets — paste yours above")

try:
    os.environ["OPENAI_API_KEY"] = userdata.get("OPENAI_API_KEY")
    print("OPENAI_API_KEY loaded from Colab Secrets ✓")
except Exception:
    os.environ["OPENAI_API_KEY"] = ""
    print("OPENAI_API_KEY not set — Condition F will be skipped")

from huggingface_hub import login
login(token=os.environ["HF_TOKEN"], add_to_git_credential=False)
print("HuggingFace login successful ✓")


# ╔══════════════════════════════════════════════════════════════════════╗
# ║  CELL 7 — Run tests (expected: 26 passed, 2 warnings)              ║
# ╚══════════════════════════════════════════════════════════════════════╝
# If you still see typeguard ImportError → re-run Cell 4 and Cell 5 ending
get_ipython().system('python -m pytest tests/ -v --tb=short 2>&1')


# ╔══════════════════════════════════════════════════════════════════════╗
# ║  CELL 8 — Generate clean data + fit anomaly filter (~3-5 min)      ║
# ╚══════════════════════════════════════════════════════════════════════╝
get_ipython().system('python scripts/generate_clean_data.py --n_episodes 200 --n_steps 50')


# ╔══════════════════════════════════════════════════════════════════════╗
# ║  CELL 9 — Quick smoke test: all 5 conditions with MockLLM (<60s)  ║
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
