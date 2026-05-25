"""
Generate clean observation data and fit the SemanticAnomalyFilter.

Usage:
    python scripts/generate_clean_data.py [--n_episodes 200] [--output_dir data/]

Steps:
  1. Run N clean (no attack) episodes with MockCAGE4Wrapper
  2. Save all observation texts to data/clean_observations.jsonl
  3. Fit SemanticAnomalyFilter on the clean observations
  4. Auto-tune threshold using a held-out 20% validation split
  5. Save fitted filter to data/filter_fitted.pkl
  6. Print: threshold, TPR, FPR
"""

import json
import logging
import os
import sys
from pathlib import Path

# Ensure project root is on sys.path regardless of invocation directory.
# Handles: python scripts/generate_clean_data.py  (from project root)
#          python /abs/path/scripts/generate_clean_data.py  (from anywhere)
_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)
# Also add cwd in case running from a subdirectory
import os as _os
_cwd = _os.getcwd()
if _cwd not in sys.path:
    sys.path.insert(0, _cwd)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def generate_poisoned_samples(n_samples: int = 200) -> list[str]:
    """Generate sample poisoned observations for threshold tuning validation."""
    from src.env.cage4_wrapper import make_env
    from src.attacks.observation_poison import ObservationPoisoner
    from src.attacks.prompt_inject import PromptInjector
    from src.attacks.comm_poison import CommPoisoner

    env = make_env(max_steps=50, seed=9999)
    attackers = [
        ObservationPoisoner(compromise_prob=0.6, false_clear_prob=0.3, seed=101),
        PromptInjector(seed=102),
        CommPoisoner(seed=103),
    ]

    poisoned = []
    ep = 0
    while len(poisoned) < n_samples:
        obs = env.reset()
        for step in range(20):
            for agent_id in env.get_agent_ids():
                obs_text = env.format_observation(agent_id, obs.get(agent_id, {}))
                attacker = attackers[ep % len(attackers)]
                poisoned_text = attacker.poison(obs_text, step, {})
                if poisoned_text != obs_text:
                    poisoned.append(poisoned_text)
                if len(poisoned) >= n_samples:
                    break
            if len(poisoned) >= n_samples:
                break
            obs, _, done, _ = env.step({a: "Monitor" for a in env.get_agent_ids()})
            if done:
                break
        ep += 1

    return poisoned[:n_samples]


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Generate clean data + fit anomaly filter")
    parser.add_argument("--n_episodes", type=int, default=200)
    parser.add_argument("--n_steps", type=int, default=50)
    parser.add_argument("--output_dir", default="data/")
    parser.add_argument("--filter_path", default="data/filter_fitted.pkl")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # ── Step 1: Generate clean observations ──────────────────────────────────
    logger.info(f"Generating {args.n_episodes} clean episodes…")

    from src.env.cage4_wrapper import make_env

    clean_obs: list[str] = []
    for ep in range(args.n_episodes):
        env = make_env(max_steps=args.n_steps, seed=ep)
        obs = env.reset()
        for step in range(args.n_steps):
            for agent_id in env.get_agent_ids():
                obs_text = env.format_observation(agent_id, obs.get(agent_id, {}))
                clean_obs.append(obs_text)
            obs, _, done, _ = env.step({a: "Monitor" for a in env.get_agent_ids()})
            if done:
                break

    logger.info(f"Collected {len(clean_obs)} clean observations.")

    # Save clean observations
    clean_path = os.path.join(args.output_dir, "clean_observations.jsonl")
    with open(clean_path, "w") as f:
        for obs in clean_obs:
            f.write(json.dumps({"obs_text": obs}) + "\n")
    logger.info(f"Saved clean observations to {clean_path}")

    # ── Step 2: Fit the filter ───────────────────────────────────────────────
    try:
        from src.defense.anomaly_filter import SemanticAnomalyFilter
    except Exception as e:
        logger.error(f"Cannot import SemanticAnomalyFilter: {e}")
        logger.error("Install: pip install sentence-transformers scikit-learn")
        return

    filt = SemanticAnomalyFilter()

    # 80/20 train/val split
    split = int(len(clean_obs) * 0.8)
    train_obs = clean_obs[:split]
    val_clean = clean_obs[split:]

    logger.info(f"Fitting on {len(train_obs)} training observations…")
    filt.fit(train_obs, save_path=args.filter_path)

    # ── Step 3: Auto-tune threshold ──────────────────────────────────────────
    logger.info("Generating poisoned validation samples for threshold tuning…")
    val_poisoned = generate_poisoned_samples(n_samples=min(len(val_clean), 500))

    logger.info(
        f"Auto-tuning threshold with {len(val_clean)} clean + "
        f"{len(val_poisoned)} poisoned validation samples…"
    )
    tune_result = filt.auto_tune_threshold(val_clean, val_poisoned, min_tpr=0.85)

    # Re-save with tuned threshold
    filt.fit(train_obs, save_path=args.filter_path)

    print("\n" + "=" * 50)
    print("ANOMALY FILTER FITTED")
    print("=" * 50)
    print(f"  Training observations: {len(train_obs)}")
    print(f"  Validation clean:      {len(val_clean)}")
    print(f"  Validation poisoned:   {len(val_poisoned)}")
    print(f"  Threshold (cosine):    {tune_result['threshold']:.4f}")
    print(f"  TPR (Catch Rate):      {tune_result['TPR']:.3f}")
    print(f"  FPR:                   {tune_result['FPR']:.3f}")
    print(f"  Filter saved to:       {args.filter_path}")
    print(f"  Filter stats:          {filt.get_stats()}")


if __name__ == "__main__":
    main()
