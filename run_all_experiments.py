"""
Run All Experiments
--------------------
Sequentially runs all 6 experimental conditions (A-F).
Saves per-condition logs and scorecards.
Prints combined comparison table.

Conditions:
  A: Baseline (no attack, no defense)
  B: Ablation — obs_poison + anomaly_filter only
  C: Ablation — obs_poison + anomaly_filter + provenance
  D: Full defense — multi_step_chain + all 3 defense layers
  E: NOVEL — adaptive white-box attacker + full defense
  F: NOVEL — cross-model (openai_llm) + multi_step_chain + full defense

Usage:
    python run_all_experiments.py --n_episodes 3  # smoke test
    python run_all_experiments.py --n_episodes 50 # full run
"""

import argparse
import json
import os
from datetime import datetime

import yaml

from src.benchmark.metrics import Scorecard
from src.benchmark.logger import save_jsonl, load_jsonl
from src.run_experiment import run_episode


CONDITION_CONFIGS = {
    "A_baseline": {
        "name": "A_baseline",
        "attack_type": None,
        "defense_layers": [],
        "agent_model": "mock_llm",
        "use_mock": True,
        "n_steps": 30,
    },
    "B_filter_only": {
        "name": "B_filter_only",
        "attack_type": "obs_poison",
        "defense_layers": ["anomaly_filter"],
        "agent_model": "mock_llm",
        "use_mock": True,
        "n_steps": 30,
        "filter_path": "data/filter_fitted.pkl",
    },
    "C_filter_provenance": {
        "name": "C_filter_provenance",
        "attack_type": "obs_poison",
        "defense_layers": ["anomaly_filter", "provenance"],
        "agent_model": "mock_llm",
        "use_mock": True,
        "n_steps": 30,
        "filter_path": "data/filter_fitted.pkl",
    },
    "D_defense_full": {
        "name": "D_defense_full",
        "attack_type": "multi_step_chain",
        "defense_layers": ["anomaly_filter", "provenance", "consistency"],
        "agent_model": "mock_llm",
        "use_mock": True,
        "n_steps": 50,
        "filter_path": "data/filter_fitted.pkl",
    },
    "E_adaptive_whitebox": {
        "name": "E_adaptive_whitebox",
        "attack_type": "adaptive",
        "defense_layers": ["anomaly_filter", "provenance", "consistency"],
        "agent_model": "mock_llm",
        "use_mock": True,
        "n_steps": 50,
        "filter_path": "data/filter_fitted.pkl",
    },
    "F_cross_model": {
        "name": "F_cross_model",
        "attack_type": "multi_step_chain",
        "defense_layers": ["anomaly_filter", "provenance", "consistency"],
        "agent_model": "openai_llm",  # GPT-4o-mini (falls back to Gemini -> Mock)
        "use_mock": False,
        "n_steps": 50,
        "filter_path": "data/filter_fitted.pkl",
    },
}


def run_all(n_episodes: int = 3, output_dir: str = "results/", real_env: bool = False):
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    all_results = {}
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Override use_mock if real_env is requested
    if real_env:
        for cond_key in CONDITION_CONFIGS:
            CONDITION_CONFIGS[cond_key]["use_mock"] = False

    print(f"\n{'='*70}")
    print(f"TacticalGuard-LLM: Running All 6 Conditions ({n_episodes} episodes each)")
    print(f"{'='*70}\n")

    from tqdm import tqdm

    for cond_key, config in CONDITION_CONFIGS.items():
        print(f"\n{'-'*60}")
        print(f"[>] Condition {cond_key}")
        print(f"  Attack: {config['attack_type']} | Defense: {config['defense_layers']}")
        print(f"  Model: {config['agent_model']}")
        print(f"{'-'*60}")

        all_logs = []
        for ep in tqdm(range(n_episodes), desc=cond_key, leave=False):
            config["env_seed"] = ep
            try:
                ep_logs = run_episode(config, episode=ep)
                all_logs.extend(ep_logs)
            except Exception as e:
                print(f"  [WARNING] Episode {ep} failed: {e}")
                continue

        if not all_logs:
            print(f"  [ERROR] No logs generated for {cond_key}")
            continue

        # Save logs
        log_file = f"results/condition_{cond_key}_logs.jsonl"
        save_jsonl(all_logs, log_file)

        # Compute scorecard
        sc = Scorecard()
        results = sc.compute(all_logs)
        all_results[cond_key] = results

        # Save scorecard
        sc.save_json(results, f"results/condition_{cond_key}_scorecard.json")
        sc.save_csv(
            {"condition": cond_key, **results},
            f"results/condition_{cond_key}_scorecard.csv"
        )

        print(f"  CAR={results['CAR']:.4f} | MTTF={results['MTTF']:.1f} | "
              f"CatchRate={results['CatchRate']:.4f} | Steps={results['total_steps']}")

    # Combined comparison table
    print(f"\n{'='*70}")
    print(f"COMBINED RESULTS COMPARISON")
    print(f"{'='*70}")

    if all_results:
        header = f"{'Condition':<25} {'CAR':>8} {'MTTF':>8} {'CatchRate':>10} {'FPR':>8} {'Lat(ms)':>8} {'AER':>8}"
        print(header)
        print("-" * len(header))
        for cond, res in all_results.items():
            print(
                f"{cond:<25} {res['CAR']:>8.4f} {res['MTTF']:>8.1f} "
                f"{res['CatchRate']:>10.4f} {res['FPR']:>8.4f} "
                f"{res['mean_latency_ms']:>8.1f} {res['AER']:>8.4f}"
            )

    # Save combined results
    combined_path = f"results/all_conditions_results_{timestamp}.json"
    with open(combined_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n[run_all_experiments] Combined results saved -> {combined_path}")

    # Generate LaTeX table
    if all_results:
        sc = Scorecard()
        latex = sc.generate_latex_table(all_results)
        os.makedirs("paper/tables", exist_ok=True)
        with open("paper/tables/results_table.tex", "w", encoding="utf-8") as f:
            f.write(latex)
        print(f"[run_all_experiments] LaTeX table saved -> paper/tables/results_table.tex")

    return all_results


def main():
    parser = argparse.ArgumentParser(description="TacticalGuard-LLM: Run All 6 Conditions")
    parser.add_argument("--n_episodes", type=int, default=3,
                        help="Episodes per condition (3=smoke test, 50=full)")
    parser.add_argument("--output_dir", default="results/")
    parser.add_argument("--real_env", action="store_true",
                        help="Force use_mock=False to use real CAGE 4 environment")
    args = parser.parse_args()

    results = run_all(n_episodes=args.n_episodes, output_dir=args.output_dir, real_env=args.real_env)
    return results


if __name__ == "__main__":
    main()
