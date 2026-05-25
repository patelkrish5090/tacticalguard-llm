"""
Run all 6 experimental conditions for the TacticalGuard-LLM paper.

Conditions:
  A: Baseline (no attack, no defense)
  B: obs_poison attack vs anomaly_filter only
  C: obs_poison attack vs filter + provenance
  D: multi_step_chain attack vs all 3 defense layers (full system)
  E: adaptive attack (white-box) vs all 3 defense layers  [NOVEL]
  F: multi_step_chain vs all 3 layers, GPT-4o-mini model  [NOVEL cross-model]

Results saved to results/condition_{A-F}_*.jsonl and *_scorecard.json.
Combined comparison table printed and LaTeX table saved.
"""

import json
import logging
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


CONDITIONS = {
    "A": {
        "name": "A_baseline",
        "attack_type": None,
        "defense_layers": [],
        "agent_model": "local_llm",
        "n_steps": 50,
        "description": "Baseline (no attack, no defense)",
    },
    "B": {
        "name": "B_filter_only",
        "attack_type": "obs_poison",
        "defense_layers": ["anomaly_filter"],
        "agent_model": "local_llm",
        "n_steps": 50,
        "description": "obs_poison attack vs anomaly_filter",
    },
    "C": {
        "name": "C_filter_provenance",
        "attack_type": "obs_poison",
        "defense_layers": ["anomaly_filter", "provenance"],
        "agent_model": "local_llm",
        "n_steps": 50,
        "description": "obs_poison attack vs filter + provenance",
    },
    "D": {
        "name": "D_full_defense",
        "attack_type": "multi_step_chain",
        "defense_layers": ["anomaly_filter", "provenance", "consistency"],
        "agent_model": "local_llm",
        "n_steps": 50,
        "description": "multi_step_chain vs full 3-layer defense",
    },
    "E": {
        "name": "E_adaptive_whitebox",
        "attack_type": "adaptive",
        "defense_layers": ["anomaly_filter", "provenance", "consistency"],
        "agent_model": "local_llm",
        "n_steps": 50,
        "description": "[NOVEL] Adaptive white-box attacker vs full defense",
    },
    "F": {
        "name": "F_cross_model",
        "attack_type": "multi_step_chain",
        "defense_layers": ["anomaly_filter", "provenance", "consistency"],
        "agent_model": "openai_llm",  # GPT-4o-mini
        "n_steps": 50,
        "description": "[NOVEL] Cross-model: multi_step_chain vs full defense (GPT-4o-mini)",
    },
}

N_EPISODES = 50  # Set lower for faster testing: 10


def run_condition(
    condition_key: str,
    config: dict,
    n_episodes: int,
    output_dir: str,
    shared_llm=None,
) -> dict:
    """Run a single experimental condition and return metrics."""
    from src.benchmark.logger import save_jsonl
    from src.benchmark.metrics import Scorecard
    from src.run_experiment import run_episode

    print(f"\n{'=' * 60}")
    print(f"Condition {condition_key}: {config['description']}")
    print(f"{'=' * 60}")

    try:
        from tqdm import tqdm

        ep_iter = tqdm(range(n_episodes), desc=f"Condition {condition_key}")
    except ImportError:
        ep_iter = range(n_episodes)

    all_logs = []
    start = time.time()

    for ep in ep_iter:
        config_ep = dict(config)
        config_ep["env_seed"] = ep
        try:
            logs = run_episode(config_ep, episode=ep, shared_llm=shared_llm)
            all_logs.extend(logs)
        except Exception as e:
            logger.error(f"Episode {ep} failed: {e}")
            continue

    elapsed = time.time() - start
    print(f"  Completed {n_episodes} episodes in {elapsed:.1f}s")

    # Save logs
    log_path = os.path.join(output_dir, f"condition_{condition_key}_logs.jsonl")
    save_jsonl(all_logs, log_path)

    # Compute metrics
    sc = Scorecard()
    metrics = sc.compute(all_logs)
    metrics["condition"] = condition_key
    metrics["description"] = config["description"]
    metrics["elapsed_s"] = elapsed

    scorecard_path = os.path.join(
        output_dir, f"condition_{condition_key}_scorecard.json"
    )
    with open(scorecard_path, "w") as f:
        json.dump(metrics, f, indent=2, default=str)

    def fmt_metric(name: str) -> str:
        value = metrics.get(name)
        return f"{value:.3f}" if isinstance(value, (int, float)) else "N/A"

    print(
        f"  CAR: {fmt_metric('CAR')}  "
        f"MTTF: {fmt_metric('MTTF')}  "
        f"CatchRate: {fmt_metric('CatchRate')}"
    )

    return metrics


def print_comparison_table(all_results: dict[str, dict]) -> None:
    """Print a formatted comparison table to stdout."""
    cols = ["CAR", "MTTF", "CatchRate", "FPR", "latency_mean_ms", "DST", "AER"]
    header = f"{'Cond':<5} {'Description':<45} " + " ".join(f"{c:>14}" for c in cols)
    print("\n" + "=" * len(header))
    print("COMBINED RESULTS TABLE")
    print("=" * len(header))
    print(header)
    print("-" * len(header))

    for cond_key in sorted(all_results.keys()):
        m = all_results[cond_key]
        desc = m.get("description", "")[:44]
        vals = []
        for c in cols:
            v = m.get(c)
            if v is None or (isinstance(v, float) and v != v):
                vals.append(f"{'--':>14}")
            elif v == float("inf"):
                vals.append(f"{'inf':>14}")
            else:
                vals.append(f"{v:>14.3f}")
        print(f"{cond_key:<5} {desc:<45} " + " ".join(vals))
    print("=" * len(header))


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Run all TacticalGuard-LLM conditions")
    parser.add_argument("--n_episodes", type=int, default=N_EPISODES)
    parser.add_argument("--output_dir", default="results/")
    parser.add_argument(
        "--conditions",
        nargs="+",
        default=list(CONDITIONS.keys()),
        help="Which conditions to run (e.g. A B C)",
    )
    parser.add_argument(
        "--skip_openai",
        action="store_true",
        help="Skip condition F (requires OPENAI_API_KEY)",
    )
    parser.add_argument(
        "--agent_model",
        choices=["local_llm"],
        help="Override conditions A-E to use local_llm",
    )
    parser.add_argument(
        "--n_steps", type=int, help="Override number of environment steps per episode"
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs("paper/tables", exist_ok=True)

    all_results: dict[str, dict] = {}
    skipped = []

    for cond_key in args.conditions:
        if cond_key not in CONDITIONS:
            logger.warning(f"Unknown condition: {cond_key}, skipping.")
            continue

        config = dict(CONDITIONS[cond_key])
        if cond_key != "F" and args.agent_model:
            config["agent_model"] = args.agent_model
        if args.n_steps:
            config["n_steps"] = args.n_steps

        # Skip openai condition if requested or no API key
        if config.get("agent_model") == "openai_llm" and (
            args.skip_openai or not os.environ.get("OPENAI_API_KEY")
        ):
            logger.warning(
                f"Skipping condition {cond_key} (openai_llm requires OPENAI_API_KEY). "
                "Set OPENAI_API_KEY env var or use --skip_openai flag."
            )
            skipped.append(cond_key)
            continue

        # Pre-load the LLM ONCE per condition (not once per episode!)
        # This avoids reloading the 8B model 50 times.
        agent_model_type = config.get("agent_model", "local_llm")
        shared_llm = None
        if agent_model_type != "openai_llm":
            from src.llm_backend.local_llm import make_llm

            print(
                f"\nPre-loading LLM once for condition {cond_key} ({agent_model_type})..."
            )
            shared_llm = make_llm(model_type=agent_model_type)
            print("LLM ready. Starting episodes...")

        try:
            metrics = run_condition(
                cond_key,
                config,
                args.n_episodes,
                args.output_dir,
                shared_llm=shared_llm,
            )
            all_results[cond_key] = metrics
        except Exception as e:
            logger.error(f"Condition {cond_key} failed: {e}", exc_info=True)
            skipped.append(cond_key)

    if not all_results:
        print("No conditions completed successfully.")
        return

    # Print comparison table
    print_comparison_table(all_results)

    # Save combined scorecard
    combined_path = os.path.join(args.output_dir, "combined_scorecard.json")
    with open(combined_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nCombined scorecard saved to: {combined_path}")

    # Generate LaTeX table
    from src.benchmark.metrics import Scorecard

    sc = Scorecard()
    latex = sc.generate_latex_table(all_results)
    latex_path = "paper/tables/results_table.tex"
    os.makedirs("paper/tables", exist_ok=True)
    with open(latex_path, "w", encoding="utf-8") as f:
        f.write(latex)
    print(f"LaTeX table saved to: {latex_path}")

    if skipped:
        print(f"\nSkipped conditions: {skipped}")


if __name__ == "__main__":
    main()
