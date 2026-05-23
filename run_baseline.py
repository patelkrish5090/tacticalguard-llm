"""
run_baseline.py
---------------
10-episode vanilla baseline runner. Logs to logs/baseline_TIMESTAMP.jsonl.
Prints summary stats.

Usage:
    python run_baseline.py --n_episodes 3 --use_mock
"""

import argparse
import json
import os
from collections import Counter
from datetime import datetime

import yaml

from src.benchmark.logger import save_jsonl
from src.benchmark.metrics import Scorecard


def main():
    parser = argparse.ArgumentParser(description="TacticalGuard-LLM Baseline Runner")
    parser.add_argument("--n_episodes", type=int, default=10)
    parser.add_argument("--use_mock", action="store_true",
                        help="Use MockLLM and MockCAGE4 (for smoke testing)")
    parser.add_argument("--config", default="configs/baseline.yaml")
    parser.add_argument("--output_dir", default="logs/")
    args = parser.parse_args()

    config = yaml.safe_load(open(args.config))
    if args.use_mock:
        config["agent_model"] = "mock_llm"
        config["use_mock"] = True

    os.makedirs(args.output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(args.output_dir, f"baseline_{timestamp}.jsonl")

    print(f"\n{'='*60}")
    print(f"TacticalGuard-LLM Baseline Run")
    print(f"Episodes: {args.n_episodes} | Model: {config.get('agent_model')}")
    print(f"{'='*60}\n")

    from src.run_experiment import run_episode
    from tqdm import tqdm

    all_logs = []
    for ep in tqdm(range(args.n_episodes), desc="Baseline episodes"):
        config["env_seed"] = ep
        ep_logs = run_episode(config, episode=ep)
        all_logs.extend(ep_logs)

    save_jsonl(all_logs, log_file)
    print(f"\n[Baseline] Saved {len(all_logs)} step logs -> {log_file}")

    # Summary statistics
    sc = Scorecard()
    results = sc.compute(all_logs)

    action_dist = Counter(s["action_parsed"] for s in all_logs)
    mean_reward = sum(s.get("reward", 0) for s in all_logs) / max(1, len(all_logs))
    mean_lat = results["mean_latency_ms"]

    print(f"\n{'='*60}")
    print(f"BASELINE SUMMARY ({args.n_episodes} episodes)")
    print(f"  Mean reward:      {mean_reward:.4f}")
    print(f"  Mean latency:     {mean_lat:.1f} ms")
    print(f"  CAR:              {results['CAR']:.4f}")
    print(f"  Action dist:      {dict(action_dist.most_common())}")
    print(f"{'='*60}")

    # Show last 3 log lines
    print(f"\nLast 3 log lines from {log_file}:")
    with open(log_file) as f:
        lines = f.readlines()
    for line in lines[-3:]:
        print(" ", line.strip()[:120])

    return results


if __name__ == "__main__":
    main()
