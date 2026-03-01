"""
CS 5374 - Adversarial Test Results Visualizer
Generates publication-ready charts from batch summary CSV data.

Usage:
    python visualize_results.py                              # Auto-find latest CSV
    python visualize_results.py batch_summary_20260225.csv   # Specific file
"""

import csv
import sys
import os
import glob
from collections import defaultdict

try:
    import matplotlib
    matplotlib.use("Agg")  # Non-interactive backend
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import numpy as np
except ImportError:
    print("matplotlib and numpy required. Install with:")
    print("  pip install matplotlib numpy")
    sys.exit(1)


def load_csv(filepath):
    """Load batch summary CSV into list of dicts."""
    rows = []
    with open(filepath, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            row["rounds"] = int(row["rounds"])
            row["max_rounds"] = int(row["max_rounds"])
            row["extracted"] = row["extracted"] == "True"
            row["avg_danger"] = float(row["avg_danger"])
            row["max_danger"] = int(row["max_danger"])
            row["partial_leaks"] = int(row["partial_leaks"])
            row["full_leaks"] = int(row["full_leaks"])
            rows.append(row)
    return rows


def aggregate_by_config(rows):
    """Group trials by attacker->target configuration."""
    configs = defaultdict(list)
    for r in rows:
        key = (r["attacker"], r["target"])
        configs[key].append(r)
    return configs


def find_csv():
    """Find the most recent batch_summary CSV."""
    candidates = glob.glob("batch_summary_*.csv")
    if not candidates:
        print("No batch_summary_*.csv found in current directory.")
        sys.exit(1)
    candidates.sort(reverse=True)
    return candidates[0]


# ============================================================
# CHART 1: Breach Rate Heatmap
# ============================================================

def plot_breach_heatmap(configs, output_dir):
    """Model-vs-model heatmap of extraction success rate."""
    attackers = sorted(set(k[0] for k in configs.keys()))
    targets = sorted(set(k[1] for k in configs.keys()))

    matrix = np.zeros((len(attackers), len(targets)))
    annotations = [['' for _ in targets] for _ in attackers]

    for i, atk in enumerate(attackers):
        for j, tgt in enumerate(targets):
            trials = configs.get((atk, tgt), [])
            if trials:
                n = len(trials)
                breaches = sum(1 for t in trials if t["extracted"])
                rate = breaches / n
                matrix[i][j] = rate
                annotations[i][j] = f"{breaches}/{n}\n({rate:.0%})"
            else:
                matrix[i][j] = -0.1  # Missing data
                annotations[i][j] = "N/A"

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(matrix, cmap="RdYlGn_r", vmin=0, vmax=1, aspect="auto")

    ax.set_xticks(range(len(targets)))
    ax.set_xticklabels(targets, fontsize=10)
    ax.set_yticks(range(len(attackers)))
    ax.set_yticklabels(attackers, fontsize=10)
    ax.set_xlabel("Target Model", fontsize=12, fontweight="bold")
    ax.set_ylabel("Attacker Model", fontsize=12, fontweight="bold")
    ax.set_title("Secret Extraction Success Rate\n(Attacker vs Target)", fontsize=14, fontweight="bold")

    for i in range(len(attackers)):
        for j in range(len(targets)):
            color = "white" if matrix[i][j] > 0.5 else "black"
            ax.text(j, i, annotations[i][j], ha="center", va="center",
                    fontsize=10, color=color, fontweight="bold")

    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Breach Rate", fontsize=11)

    plt.tight_layout()
    path = os.path.join(output_dir, "breach_heatmap.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


# ============================================================
# CHART 2: Danger Score Comparison
# ============================================================

def plot_danger_scores(configs, output_dir):
    """Bar chart comparing avg and max danger scores per config."""
    labels = []
    avg_dangers = []
    max_dangers = []

    for (atk, tgt), trials in sorted(configs.items()):
        label = f"{atk}\n→ {tgt}"
        labels.append(label)
        avg_dangers.append(sum(t["avg_danger"] for t in trials) / len(trials))
        max_dangers.append(max(t["max_danger"] for t in trials))

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 6))
    bars1 = ax.bar(x - width/2, avg_dangers, width, label="Avg Danger", color="#3498db", alpha=0.85)
    bars2 = ax.bar(x + width/2, max_dangers, width, label="Max Danger", color="#e74c3c", alpha=0.85)

    ax.set_ylabel("Danger Score (0-10)", fontsize=12)
    ax.set_title("Danger Scores by Model Configuration", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.legend(fontsize=11)
    ax.set_ylim(0, 11)
    ax.axhline(y=5, color="orange", linestyle="--", alpha=0.5, label="Medium Risk")
    ax.grid(axis="y", alpha=0.3)

    # Value labels on bars
    for bar in bars1:
        h = bar.get_height()
        if h > 0:
            ax.text(bar.get_x() + bar.get_width()/2, h + 0.15, f"{h:.1f}",
                    ha="center", va="bottom", fontsize=8)
    for bar in bars2:
        h = bar.get_height()
        if h > 0:
            ax.text(bar.get_x() + bar.get_width()/2, h + 0.15, f"{int(h)}",
                    ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    path = os.path.join(output_dir, "danger_scores.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


# ============================================================
# CHART 3: Leak Events by Target Model
# ============================================================

def plot_leak_breakdown(configs, output_dir):
    """Stacked bar chart showing leak types per target model."""
    targets = sorted(set(k[1] for k in configs.keys()))

    no_leaks = []
    partial_leaks = []
    full_leaks = []

    for tgt in targets:
        total_rounds = 0
        p_leaks = 0
        f_leaks = 0
        for (atk, t), trials in configs.items():
            if t != tgt:
                continue
            for trial in trials:
                total_rounds += trial["rounds"]
                p_leaks += trial["partial_leaks"]
                f_leaks += trial["full_leaks"]
        no_leaks.append(total_rounds - p_leaks - f_leaks)
        partial_leaks.append(p_leaks)
        full_leaks.append(f_leaks)

    x = np.arange(len(targets))
    width = 0.5

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.bar(x, no_leaks, width, label="No Leak", color="#2ecc71", alpha=0.85)
    ax.bar(x, partial_leaks, width, bottom=no_leaks, label="Partial Leak", color="#f39c12", alpha=0.85)
    bottoms = [n + p for n, p in zip(no_leaks, partial_leaks)]
    ax.bar(x, full_leaks, width, bottom=bottoms, label="Full Leak", color="#e74c3c", alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(targets, fontsize=11)
    ax.set_ylabel("Total Rounds", fontsize=12)
    ax.set_title("Leak Events by Target Model\n(Aggregated Across All Attackers)", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    path = os.path.join(output_dir, "leak_breakdown.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


# ============================================================
# CHART 4: Target Model Resilience Radar
# ============================================================

def plot_resilience_radar(configs, output_dir):
    """Radar/spider chart showing multi-dimensional resilience per target."""
    targets = sorted(set(k[1] for k in configs.keys()))

    # Metrics per target (all normalized 0-1 where 1 = more resilient)
    metrics = {tgt: {} for tgt in targets}

    for tgt in targets:
        all_trials = []
        for (atk, t), trials in configs.items():
            if t == tgt:
                all_trials.extend(trials)

        if not all_trials:
            continue

        n = len(all_trials)
        breach_rate = sum(1 for t in all_trials if t["extracted"]) / n
        avg_danger = sum(t["avg_danger"] for t in all_trials) / n
        max_danger = max(t["max_danger"] for t in all_trials)
        partial_rate = sum(t["partial_leaks"] for t in all_trials) / sum(t["rounds"] for t in all_trials)

        # Invert so higher = more resilient
        metrics[tgt] = {
            "Breach\nResistance": 1 - breach_rate,
            "Low Avg\nDanger": 1 - (avg_danger / 10),
            "Low Max\nDanger": 1 - (max_danger / 10),
            "Leak\nResistance": 1 - partial_rate,
            "Rounds\nSurvived": sum(t["rounds"] for t in all_trials) / (n * all_trials[0]["max_rounds"]),
        }

    categories = list(list(metrics.values())[0].keys())
    N = len(categories)
    angles = [n / float(N) * 2 * 3.14159 for n in range(N)]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    colors = ["#3498db", "#e74c3c", "#2ecc71"]

    for idx, tgt in enumerate(targets):
        values = [metrics[tgt][cat] for cat in categories]
        values += values[:1]
        ax.plot(angles, values, "o-", linewidth=2, label=tgt, color=colors[idx % len(colors)])
        ax.fill(angles, values, alpha=0.15, color=colors[idx % len(colors)])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=10)
    ax.set_ylim(0, 1.1)
    ax.set_title("Target Model Resilience Profile\n(Higher = More Resilient)", fontsize=14, fontweight="bold", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=11)

    plt.tight_layout()
    path = os.path.join(output_dir, "resilience_radar.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


# ============================================================
# MAIN
# ============================================================

def main():
    if len(sys.argv) > 1:
        csv_file = sys.argv[1]
    else:
        csv_file = find_csv()

    print(f"Loading: {csv_file}")
    rows = load_csv(csv_file)
    print(f"Loaded {len(rows)} trial results")

    configs = aggregate_by_config(rows)
    print(f"Found {len(configs)} model configurations\n")

    output_dir = "charts"
    os.makedirs(output_dir, exist_ok=True)

    print("Generating charts...")
    plot_breach_heatmap(configs, output_dir)
    plot_danger_scores(configs, output_dir)
    plot_leak_breakdown(configs, output_dir)
    plot_resilience_radar(configs, output_dir)

    print(f"\nAll charts saved to: {output_dir}/")
    print("Charts generated:")
    print("  - breach_heatmap.png    : Which model combos succeed/fail")
    print("  - danger_scores.png     : Risk levels per configuration")
    print("  - leak_breakdown.png    : Leak types by target model")
    print("  - resilience_radar.png  : Multi-dimensional resilience profile")


if __name__ == "__main__":
    main()
