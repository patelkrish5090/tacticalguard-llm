"""
Baseline runner: vanilla LLM defending CAGE 4 with no attacks or defenses.

Usage:
    python run_baseline.py [--n_episodes 10] [--output_dir results/]
"""

import argparse
import json
import logging
import os
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="TacticalGuard-LLM Baseline")
    parser.add_argument("--n_episodes", type=int, default=10)
    parser.add_argument("--n_steps", type=int, default=50)
    parser.add_argument("--output_dir", default="results/")
    parser.add_argument("--agent_model", default="mock",
                        help="mock | local_llm | openai_llm")
    args = parser.parse_args()

    config = {
        "name": "baseline",
        "attack_type": None,
        "defense_layers": [],
        "agent_model": args.agent_model,
        "n_steps": args.n_steps,
    }

    from src.run_experiment import run_episode
    from src.benchmark.logger import make_log_path, save_jsonl
    from src.benchmark.metrics import Scorecard

    all_logs = []
    try:
        from tqdm import tqdm
        ep_iter = tqdm(range(args.n_episodes), desc="Baseline episodes")
    except ImportError:
        ep_iter = range(args.n_episodes)

    for ep in ep_iter:
        config["env_seed"] = ep
        logs = run_episode(config, episode=ep)
        all_logs.extend(logs)

    os.makedirs(args.output_dir, exist_ok=True)
    log_path = make_log_path(args.output_dir, "baseline")
    save_jsonl(all_logs, log_path)

    # Summary
    rewards = [r["reward"] for r in all_logs if r.get("reward") is not None]
    latencies = [r["latency_ms"] for r in all_logs if r.get("latency_ms") is not None]
    actions = Counter(r["action_parsed"] for r in all_logs)

    sc = Scorecard()
    metrics = sc.compute(all_logs)

    print("\n" + "=" * 60)
    print("BASELINE RESULTS")
    print("=" * 60)
    print(f"  Episodes: {args.n_episodes}")
    print(f"  Total steps: {len(all_logs)}")
    print(f"  Mean reward: {sum(rewards)/len(rewards):.3f}" if rewards else "  Mean reward: N/A")
    print(f"  Mean latency: {sum(latencies)/len(latencies):.1f}ms" if latencies else "  Mean latency: N/A")
    print(f"  Action distribution:")
    for action, count in actions.most_common():
        pct = 100 * count / len(all_logs)
        print(f"    {action:<15} {count:>5} ({pct:.1f}%)")
    print(f"\nMetrics:")
    print(json.dumps(metrics, indent=2))
    print(f"\nLogs saved to: {log_path}")

    # Show last 3 log lines
    print("\nLast 3 log records:")
    for record in all_logs[-3:]:
        print(json.dumps({
            k: v for k, v in record.items()
            if k in {"episode", "step", "agent_id", "action_parsed", "reward", "latency_ms"}
        }))


if __name__ == "__main__":
    main()
