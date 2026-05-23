"""
Generate Clean Data for Anomaly Filter Training
------------------------------------------------
Runs N clean (no attack) episodes and saves all observation texts.
Fits SemanticAnomalyFilter on this data.
Runs auto_tune_threshold() on 20% validation split.

Usage:
    python scripts/generate_clean_data.py --n_episodes 20
    (Use 200 for full training, 20 for smoke test)
"""

import argparse
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.env.cage4_wrapper import MockCAGE4Wrapper
from src.defense.anomaly_filter import SemanticAnomalyFilter
from src.attacks.observation_poison import ObservationPoisoner
from src.benchmark.logger import save_jsonl


def collect_clean_observations(n_episodes: int = 20, n_steps: int = 30) -> list:
    """Run clean episodes and collect all observation texts."""
    all_obs = []
    env = MockCAGE4Wrapper(max_steps=n_steps)
    agent_ids = env.get_agent_ids()

    print(f"[generate_clean_data] Collecting {n_episodes} episodes × {n_steps} steps...")

    for ep in range(n_episodes):
        obs = env.reset()
        for step in range(n_steps):
            for agent_id in agent_ids:
                raw_obs = obs.get(agent_id, {})
                obs_text = env.format_observation(agent_id, raw_obs, step=step)
                all_obs.append({
                    "episode": ep,
                    "step": step,
                    "agent_id": agent_id,
                    "obs_text": obs_text,
                })
            actions = {aid: "Monitor" for aid in agent_ids}
            obs, _, done, _ = env.step(actions)
            if done:
                break

    return all_obs


def generate_poisoned_observations(n_samples: int = 100) -> list:
    """Generate poisoned observations for validation split."""
    env = MockCAGE4Wrapper(max_steps=30)
    poisoner = ObservationPoisoner(compromise_prob=0.5, false_clear_prob=0.3)
    agent_ids = env.get_agent_ids()
    poisoned_texts = []

    obs = env.reset()
    for step in range(n_samples // len(agent_ids) + 1):
        for agent_id in agent_ids:
            raw_obs = obs.get(agent_id, {})
            obs_text = env.format_observation(agent_id, raw_obs, step=step)
            poisoned = poisoner.poison(obs_text, step, {"step": step})
            if poisoned != obs_text:
                poisoned_texts.append(poisoned)
        actions = {aid: "Monitor" for aid in agent_ids}
        obs, _, done, _ = env.step(actions)
        if done:
            break
        if len(poisoned_texts) >= n_samples:
            break

    return poisoned_texts[:n_samples]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_episodes", type=int, default=20,
                        help="Number of clean episodes (200 for full, 20 for smoke test)")
    parser.add_argument("--output_dir", default="data/")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # 1. Collect clean observations
    clean_records = collect_clean_observations(n_episodes=args.n_episodes, n_steps=30)
    clean_texts = [r["obs_text"] for r in clean_records]

    # Save raw observations
    obs_path = os.path.join(args.output_dir, "clean_observations.jsonl")
    save_jsonl(clean_records, obs_path)
    print(f"[generate_clean_data] Saved {len(clean_records)} clean obs -> {obs_path}")

    # 2. Train/val split (80/20)
    n_train = int(0.8 * len(clean_texts))
    train_texts = clean_texts[:n_train]
    val_clean_texts = clean_texts[n_train:]

    # 3. Fit anomaly filter
    filter_path = os.path.join(args.output_dir, "filter_fitted.pkl")
    filt = SemanticAnomalyFilter(contamination=0.05, sim_threshold=0.82)
    filt.fit(train_texts, save_path=filter_path)

    # 4. Generate poisoned validation set
    n_val_poisoned = max(20, len(val_clean_texts) // 2)
    val_poisoned_texts = generate_poisoned_observations(n_samples=n_val_poisoned)
    print(f"[generate_clean_data] Generated {len(val_poisoned_texts)} poisoned validation samples")

    # 5. Auto-tune threshold
    if val_clean_texts and val_poisoned_texts:
        tune_result = filt.auto_tune_threshold(
            val_clean=val_clean_texts[:50],
            val_poisoned=val_poisoned_texts[:50],
        )
        print(f"[generate_clean_data] Threshold tuning result: {tune_result}")

        # Re-save with updated threshold
        filt.fit(train_texts, save_path=filter_path)

    # 6. Quick sanity check
    stats = filt.get_stats()
    print(f"\n[generate_clean_data] Filter stats: {stats}")
    print(f"\n[+] Anomaly filter trained and saved -> {filter_path}")
    print(f"[+] Threshold: {filt.threshold:.4f}")


if __name__ == "__main__":
    main()
