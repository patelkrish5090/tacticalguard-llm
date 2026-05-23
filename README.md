# TacticalGuard-LLM

**Adversarial Robustness of LLM Cyber-Defense Agents in CAGE 4**
*MILCOM 2026 Research System*

---

## Overview

TacticalGuard-LLM benchmarks LLM-based autonomous cyber-defense (ACD) agents under multi-vector adversarial attacks in the CAGE 4 military simulation, and defends them with a 3-layer pipeline.

**Key Distinction:** We attack *against* LLM defenders (Blue agents), not *with* LLMs against RL defenders. This complements Castro et al. (2505.04843) who showed LLMs can act as ACD agents, but did not evaluate adversarial robustness.

---

## Project Structure

```
tacticalguard-llm/
├── src/
│   ├── env/
│   │   ├── cage4_wrapper.py        # CAGE 4 + MockCAGE4 environment
│   │   └── action_space.py         # Action parser (regex + fuzzy)
│   ├── llm_backend/
│   │   ├── local_llm.py            # Llama-3.1-8B-Instruct (4-bit) + MockLLM
│   │   └── openai_llm.py           # GPT-4o-mini + Gemini fallback
│   ├── attacks/
│   │   ├── observation_poison.py   # False COMPROMISED / false clear injection
│   │   ├── comm_poison.py          # Fake teammate message injection
│   │   ├── reward_hack.py          # Traffic camouflage for compromised hosts
│   │   ├── prompt_inject.py        # Direct instruction injection
│   │   └── multi_step_chain.py     # NOVEL: 3-phase APT-style attack chain
│   ├── defense/
│   │   ├── anomaly_filter.py       # Semantic filter (MiniLM + IsolationForest)
│   │   ├── provenance_prompt.py    # Reliability-tagged prompts
│   │   └── consistency_guard.py    # Self-consistency voting + mismatch detection
│   ├── benchmark/
│   │   ├── metrics.py              # 7 metrics: CAR, MTTF, CatchRate, FPR, Latency, DST, AER
│   │   ├── adaptive_attacker.py    # NOVEL: white-box/black-box adaptive attacker
│   │   └── logger.py               # JSONL step logger
│   └── run_experiment.py           # Full episode loop
├── configs/                        # YAML configs for all 6 conditions
├── scripts/generate_clean_data.py  # Train anomaly filter on clean data
├── run_baseline.py                 # Quick baseline runner
├── run_all_experiments.py          # All 6 conditions (A-F)
├── notebooks/analysis.ipynb        # 7-figure analysis notebook
└── tests/                          # pytest unit tests
```

---

## Quick Start

### 1. Set up virtual environment

```bash
cd tacticalguard-llm
python -m venv venv
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

pip install -r requirements.txt
```

### 2. (Optional) Install CAGE 4

```bash
git clone https://github.com/cage-challenge/cage-challenge-4
pip install -e cage-challenge-4/
```

Without CAGE 4, all scripts use `MockCAGE4Wrapper` automatically.

### 3. Set API keys (optional)

```bash
# For GPT-4o-mini (Condition F cross-model):
export OPENAI_API_KEY=your_key

# For Gemini fallback:
export GOOGLE_API_KEY=your_key

# For Llama-3.1-8B (Phase 1 LocalLLM):
export HF_TOKEN=your_hf_token
```

### 4. Train the anomaly filter

```bash
# Smoke test (20 episodes, fast):
python scripts/generate_clean_data.py --n_episodes 20

# Full training (200 episodes, for GPU runs):
python scripts/generate_clean_data.py --n_episodes 200
```

### 5. Run a quick smoke test

```bash
# Import check
python -c "from src.env.cage4_wrapper import MockCAGE4Wrapper; print('OK')"

# 3-episode baseline smoke test
python run_baseline.py --n_episodes 3 --use_mock

# Full pipeline smoke test (2 episodes)
python src/run_experiment.py --config configs/defense_full.yaml --n_episodes 2 --use_mock
```

### 6. Run unit tests

```bash
python -m pytest tests/test_attacks.py -v
python -m pytest tests/test_metrics.py -v
```

### 7. Run all 6 experimental conditions

```bash
# Smoke test (3 episodes per condition, ~5 min):
python run_all_experiments.py --n_episodes 3

# Full run (50 episodes per condition, ~8 hrs on GPU):
python run_all_experiments.py --n_episodes 50
```

### 8. Analyze results

```bash
jupyter notebook notebooks/analysis.ipynb
```

---

## Experimental Conditions

| Condition | Attack | Defense | Novel? |
|-----------|--------|---------|--------|
| A | None | None | — |
| B | Obs. Poison | Anomaly Filter | — |
| C | Obs. Poison | Filter + Provenance | — |
| D | Multi-Step Chain | All 3 Layers | ✓ (attack) |
| E | Adaptive (White-box) | All 3 Layers | ✓ (attack) |
| F | Multi-Step Chain | All 3 Layers | ✓ (cross-model GPT) |

---

## Metrics

| Metric | Description | Direction |
|--------|-------------|-----------|
| **CAR** | Catastrophic Action Rate | ↓ lower is better |
| **MTTF** | Mean Time To First Failure | ↑ higher is better |
| **CatchRate** | Anomaly filter TPR | ↑ higher is better |
| **FPR** | False Positive Rate | ↓ lower is better |
| **Latency** | Mean decision latency (ms) | ↓ lower is better |
| **DST** | Defense Survival Time (phase) | ↑ higher is better |
| **AER** | Adaptive Evasion Rate | ↓ lower is better |

DST and AER are novel metrics introduced in this paper.

---

## Hardware Requirements

| Mode | Requirement |
|------|-------------|
| MockLLM (default) | CPU only, any machine |
| LocalLLM (Llama-3.1-8B 4-bit) | CUDA GPU ≥12 GB VRAM |
| OpenAI / Gemini | API key, internet |

MockLLM mode is scientifically valid for pipeline demonstration.
Full GPU results require H100/A100/T4 (≥12 GB VRAM).

---

## Citation

```bibtex
@inproceedings{tacticalguard2026,
  title={TacticalGuard-LLM: Adversarial Robustness of LLM Cyber-Defense Agents in CAGE 4},
  author={...},
  booktitle={IEEE MILCOM 2026},
  year={2026}
}
```
