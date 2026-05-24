# TacticalGuard-LLM

**Adversarial Robustness Benchmarking for LLM-Based Tactical Cyber-Defense Agents**

Research system for MILCOM 2026. Attacks and defends LLM-based cyber-defense agents in the CAGE 4 military simulation.

## Key Contributions
1. First adversarial robustness benchmark for LLM-based ACD agents
2. Novel multi-step attack chain modeling APT-style temporal strategy
3. 3-layer defense with adaptive-aware anomaly filter
4. Cross-model transfer analysis (LLaMA vs GPT-4o-mini)

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# (Optional) Install CAGE 4
git clone https://github.com/cage-challenge/cage-challenge-4.git
pip install -e cage-challenge-4/

# Generate clean observation data + fit anomaly filter
python scripts/generate_clean_data.py

# Run baseline
python run_baseline.py

# Run all 6 experimental conditions
python run_all_experiments.py
```

## Project Structure
```
src/
├── env/            - CAGE 4 wrapper + mock environment
├── llm_backend/    - Local LLM (LLaMA 4-bit) + OpenAI fallback
├── attacks/        - 6 attack vectors including novel multi-step chain
├── defense/        - 3-layer defense pipeline
└── benchmark/      - Metrics, adaptive attacker, logger
configs/            - YAML experiment configs
data/               - Clean observations + fitted filter
results/            - Experiment logs + scorecards
scripts/            - Data generation utilities
notebooks/          - Analysis notebook
```

## Experimental Conditions
| Condition | Attack | Defense | Model |
|-----------|--------|---------|-------|
| A | None | None | LLaMA |
| B | obs_poison | anomaly_filter | LLaMA |
| C | obs_poison | filter+provenance | LLaMA |
| D | multi_step_chain | All 3 layers | LLaMA |
| E (NOVEL) | adaptive (white-box) | All 3 layers | LLaMA |
| F (NOVEL) | multi_step_chain | All 3 layers | GPT-4o-mini |

## Threat Model
The attacker controls the **observation text pipeline** only — NOT model weights or simulator internals. This is the realistic deployment threat: a compromised sensor or network tap that injects malicious content into the LLM's context window.
