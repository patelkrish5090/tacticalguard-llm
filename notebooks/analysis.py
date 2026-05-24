# TacticalGuard-LLM Results Analysis
# Jupyter Notebook (Python script format — run as notebook or script)

# ── Cell 1: Imports ───────────────────────────────────────────────────────────
import json
import os
import sys
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(".").resolve()))

import numpy as np
import pandas as pd

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    PLOTS_AVAILABLE = True
except ImportError:
    PLOTS_AVAILABLE = False
    print("matplotlib/seaborn not installed. Install: pip install matplotlib seaborn")

# Military-style dark theme
if PLOTS_AVAILABLE:
    sns.set_theme(style="darkgrid")
    MILITARY_PALETTE = ["#8B7355", "#D4AF37", "#556B2F", "#8B4513", "#2F4F4F", "#B8860B"]
    plt.rcParams.update({
        "figure.facecolor": "#1a1a2e",
        "axes.facecolor": "#16213e",
        "axes.edgecolor": "#D4AF37",
        "axes.labelcolor": "#D4AF37",
        "text.color": "#e0e0e0",
        "xtick.color": "#e0e0e0",
        "ytick.color": "#e0e0e0",
        "grid.color": "#2a2a4a",
        "figure.figsize": (10, 6),
        "font.family": "monospace",
    })

RESULTS_DIR = "results/"
FIGURES_DIR = "results/figures/"
os.makedirs(FIGURES_DIR, exist_ok=True)

CONDITION_LABELS = {
    "A": "A: Baseline",
    "B": "B: Filter Only",
    "C": "C: Filter+Prov",
    "D": "D: Full Defense",
    "E": "E: Adaptive (WB)",
    "F": "F: Cross-Model",
}

# ── Cell 2: Load Results ──────────────────────────────────────────────────────
def load_all_results():
    """Load all condition scorecards into a DataFrame."""
    records = []
    for cond_key in "ABCDEF":
        path = os.path.join(RESULTS_DIR, f"condition_{cond_key}_scorecard.json")
        if os.path.exists(path):
            with open(path) as f:
                data = json.load(f)
            data["condition_key"] = cond_key
            data["condition_label"] = CONDITION_LABELS.get(cond_key, cond_key)
            records.append(data)

    if not records:
        # Try combined scorecard
        combined_path = os.path.join(RESULTS_DIR, "combined_scorecard.json")
        if os.path.exists(combined_path):
            with open(combined_path) as f:
                combined = json.load(f)
            for key, data in combined.items():
                data["condition_key"] = key
                data["condition_label"] = CONDITION_LABELS.get(key, key)
                records.append(data)

    if not records:
        print("No result files found. Run run_all_experiments.py first.")
        return pd.DataFrame()

    return pd.DataFrame(records)

df = load_all_results()
if not df.empty:
    print("Loaded conditions:", df["condition_key"].tolist())
    print(df[["condition_key", "CAR", "MTTF", "CatchRate", "FPR", "latency_mean_ms"]].to_string(index=False))

# ── Cell 3: CAR Comparison Bar Chart ─────────────────────────────────────────
def plot_car(df):
    if df.empty or not PLOTS_AVAILABLE:
        print("No data or matplotlib not available.")
        return

    fig, ax = plt.subplots(figsize=(10, 5))
    colors = MILITARY_PALETTE[:len(df)]
    bars = ax.bar(
        df["condition_label"], df["CAR"].fillna(0),
        color=colors, edgecolor="#D4AF37", linewidth=1.2
    )
    ax.set_title("Catastrophic Action Rate (CAR) by Condition\n(lower is better)",
                 fontsize=14, color="#D4AF37", pad=15)
    ax.set_ylabel("CAR", color="#D4AF37")
    ax.set_ylim(0, max(df["CAR"].fillna(0).max() * 1.2, 0.1))

    for bar, val in zip(bars, df["CAR"].fillna(0)):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
                f"{val:.3f}", ha="center", va="bottom", fontsize=10, color="#D4AF37")

    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "car_comparison.pdf")
    plt.savefig(path, dpi=300, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.savefig(path.replace(".pdf", ".png"), dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.show()
    print(f"Saved: {path}")

plot_car(df)

# ── Cell 4: MTTF Comparison ───────────────────────────────────────────────────
def plot_mttf(df):
    if df.empty or not PLOTS_AVAILABLE:
        return
    mttf_df = df[df["MTTF"].notna() & (df["MTTF"] != float("inf"))]
    if mttf_df.empty:
        print("No MTTF data available (only computed for attacked conditions).")
        return

    fig, ax = plt.subplots(figsize=(10, 5))
    colors = MILITARY_PALETTE[:len(mttf_df)]
    bars = ax.bar(mttf_df["condition_label"], mttf_df["MTTF"],
                  color=colors, edgecolor="#D4AF37", linewidth=1.2)
    ax.set_title("Mean Time To First Failure (MTTF)\n(higher is better — defense delays failure)",
                 fontsize=13, color="#D4AF37", pad=15)
    ax.set_ylabel("Steps to First Failure", color="#D4AF37")
    for bar, val in zip(bars, mttf_df["MTTF"]):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                f"{val:.1f}", ha="center", va="bottom", fontsize=10, color="#D4AF37")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "mttf_comparison.pdf")
    plt.savefig(path, dpi=300, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.show()
    print(f"Saved: {path}")

plot_mttf(df)

# ── Cell 5: Multi-step Chain Phase Analysis ───────────────────────────────────
def plot_phase_analysis():
    """Load multi-step chain logs and plot per-phase action distributions."""
    if not PLOTS_AVAILABLE:
        return

    log_files = [
        f for f in os.listdir(RESULTS_DIR)
        if "condition_D" in f and f.endswith(".jsonl")
    ]
    if not log_files:
        print("No condition D logs found. Run experiment D first.")
        return

    from src.benchmark.logger import load_jsonl
    logs = load_jsonl(os.path.join(RESULTS_DIR, log_files[0]))

    phase_actions: dict[int, list] = {1: [], 2: [], 3: []}
    for record in logs:
        phase = record.get("attack_phase")
        if phase in phase_actions:
            phase_actions[phase].append(record.get("action_parsed", "Monitor"))

    from collections import Counter
    ACTIONS = ["Monitor", "Analyse", "Remove", "Restore", "DeployDecoy",
               "BlockTraffic", "AllowTraffic"]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    phase_labels = {1: "Phase 1:\nConfidence Erosion",
                    2: "Phase 2:\nFalse Normalization",
                    3: "Phase 3:\nDecisive Strike"}

    for i, (phase, actions) in enumerate(phase_actions.items()):
        if not actions:
            axes[i].text(0.5, 0.5, "No data", ha="center", transform=axes[i].transAxes,
                        color="#D4AF37")
            continue
        counts = Counter(actions)
        vals = [counts.get(a, 0) for a in ACTIONS]
        bars = axes[i].bar(ACTIONS, vals, color=MILITARY_PALETTE[:len(ACTIONS)],
                           edgecolor="#D4AF37", linewidth=0.8)
        axes[i].set_title(phase_labels[phase], color="#D4AF37", fontsize=11)
        axes[i].set_ylabel("Action Count" if i == 0 else "")
        axes[i].tick_params(axis="x", rotation=45)

    fig.suptitle("Action Distribution per Attack Phase (Multi-Step Chain)",
                 fontsize=13, color="#D4AF37", y=1.02)
    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "phase_analysis.pdf")
    plt.savefig(path, dpi=300, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.show()
    print(f"Saved: {path}")

plot_phase_analysis()

# ── Cell 6: Adaptive Attacker Evasion Rate ────────────────────────────────────
def plot_adaptive_evasion(df):
    if df.empty or not PLOTS_AVAILABLE:
        return
    aer_df = df[df["AER"].notna() & (df["AER"] > 0)]
    if aer_df.empty:
        print("No AER data. Condition E (adaptive) must be run.")
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(aer_df["condition_label"], aer_df["AER"],
           color=["#D4AF37", "#8B7355"][:len(aer_df)],
           edgecolor="#D4AF37", linewidth=1.2)
    ax.axhline(y=0.5, color="#FF6B6B", linestyle="--", alpha=0.7, label="50% baseline")
    ax.set_title("Adaptive Evasion Rate (AER)\n(higher = attacker more successful)",
                 color="#D4AF37", fontsize=13)
    ax.set_ylabel("Evasion Rate", color="#D4AF37")
    ax.set_ylim(0, 1.0)
    ax.legend(loc="upper right")
    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "adaptive_evasion.pdf")
    plt.savefig(path, dpi=300, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.show()
    print(f"Saved: {path}")

plot_adaptive_evasion(df)

# ── Cell 7: Cross-Model Transfer Comparison ───────────────────────────────────
def plot_cross_model(df):
    if df.empty or not PLOTS_AVAILABLE:
        return

    target_conds = ["D", "F"]
    sub = df[df["condition_key"].isin(target_conds)]
    if len(sub) < 2:
        print("Need both condition D (LLaMA) and F (GPT-4o-mini) for transfer plot.")
        return

    metrics = ["CAR", "CatchRate", "FPR", "latency_mean_ms"]
    x = np.arange(len(metrics))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 5))
    for i, (_, row) in enumerate(sub.iterrows()):
        vals = [row.get(m, 0) or 0 for m in metrics]
        bars = ax.bar(x + i * width, vals, width,
                      label=row["condition_label"],
                      color=MILITARY_PALETTE[i * 2],
                      edgecolor="#D4AF37", linewidth=0.8)

    ax.set_xticks(x + width / 2)
    ax.set_xticklabels(metrics, rotation=15)
    ax.set_title("Cross-Model Transfer: LLaMA vs GPT-4o-mini",
                 color="#D4AF37", fontsize=13)
    ax.set_ylabel("Metric Value", color="#D4AF37")
    ax.legend()
    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "cross_model_transfer.pdf")
    plt.savefig(path, dpi=300, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.show()
    print(f"Saved: {path}")

plot_cross_model(df)

# ── Cell 8: LaTeX Table ───────────────────────────────────────────────────────
def print_latex_table(df):
    if df.empty:
        return
    from src.benchmark.metrics import Scorecard

    # Rebuild results dict
    results = {}
    for _, row in df.iterrows():
        cond = row.get("condition_key", row.get("condition", "?"))
        results[cond] = row.to_dict()

    sc = Scorecard()
    latex = sc.generate_latex_table(results)
    print("\n" + "=" * 60)
    print("LaTeX Table (copy to paper):")
    print("=" * 60)
    print(latex)

print_latex_table(df)
