#!/usr/bin/env python3
"""
Multi-seed visualization of analysis results: reward decomposition, KL divergence,
parameter distance, action difference, log-std shift, KL-vs-return scatter.

Data sources:
  - results/reward_decomp.csv, reward_decomp_98.csv, reward_decomp_5075.csv
  - KL / param / action data hardcoded from analyze_kl.py output (3 seeds)

Output:
  - results/fig_reward_decomp.png      (multi-seed mean + individual dots)
  - results/fig_kl_divergence.png      (multi-seed grouped bars + dots)
  - results/fig_kl_vs_return.png       (NEW: KL-return decoupling scatter)
  - results/fig_param_distance.png     (multi-seed)
  - results/fig_action_diff.png        (multi-seed)
  - results/fig_logstd_shift.png       (multi-seed)
  - results/fig_return_vs_success.png  (multi-seed)
  - results/fig_overview.png           (4-panel summary)
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.patches import Patch
from pathlib import Path

# ── Palette ──────────────────────────────────────────────────────────────────
PALETTE = {
    "PPO":      "#eb6834",
    "GRPO":     "#2a78d6",
    "PPO-v3":   "#eda100",
    "GRPO-v3":  "#1baf7a",
}
COMP_COLORS = {
    "door_openness":    "#eb6834",
    "passage":          "#2a78d6",
    "others":           "#c3c2b7",
}
OUT_DIR = Path(__file__).resolve().parent / "results"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Global style ─────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica", "sans-serif"],
    "font.size": 10, "axes.titlesize": 12, "axes.labelsize": 11,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": 0.8, "axes.edgecolor": "#c3c2b7",
    "xtick.color": "#898781", "ytick.color": "#898781",
    "grid.color": "#e1e0d9", "grid.linewidth": 0.5,
    "legend.frameon": True, "legend.framealpha": 0.9,
    "legend.edgecolor": "#e1e0d9", "legend.fontsize": 9,
    "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb",
    "savefig.dpi": 200, "savefig.bbox": "tight", "savefig.facecolor": "#fcfcfb",
})

# ── Multi-seed hardcoded data (from analyze_kl.py / analyze_reward.py) ────────

SEEDS = [42, 98, 5075]

# KL(π_finetuned || π_baseline) mean nats
KL_MULTISEED = {
    "Misaligned Reward": {
        "PPO":  {42: 25.36, 98: 33.17, 5075: 39.15},
        "GRPO": {42: 19.42, 98: 11.67, 5075: 3.84},
    },
    "Sparse Reward": {
        "PPO":  {42: 47.36, 98: 3.68, 5075: 2.82},
        "GRPO": {42: 19.64, 98: 4.20, 5075: 25.23},
    },
}

# ||μ_finetune - μ_baseline|| mean L2
ACTION_DIFF_MULTISEED = {
    "Misaligned Reward": {
        "PPO":  {42: 2.21, 98: 2.56, 5075: 2.77},
        "GRPO": {42: 1.94, 98: 1.51, 5075: 0.88},
    },
    "Sparse Reward": {
        "PPO":  {42: 3.02, 98: 0.84, 5075: 0.74},
        "GRPO": {42: 1.71, 98: 0.82, 5075: 1.94},
    },
}

# Mean Δ log_σ
LOGSTD_MULTISEED = {
    "Misaligned Reward": {
        "PPO":  {42: -0.0524, 98: -0.0591, 5075: -0.0769},
        "GRPO": {42: -0.0235, 98: -0.0197, 5075: -0.0051},
    },
    "Sparse Reward": {
        "PPO":  {42: -0.0540, 98: -0.0050, 5075: -0.0041},
        "GRPO": {42: +0.0018, 98: +0.0012, 5075: +0.0022},
    },
}

# Total param L2 distance
PARAM_MULTISEED = {
    "Misaligned Reward": {
        "PPO":  {42: 2.73, 98: 3.03, 5075: 3.35},
        "GRPO": {42: 2.39, 98: 1.96, 5075: 1.14},
    },
    "Sparse Reward": {
        "PPO":  {42: 3.37, 98: 0.96, 5075: 0.96},
        "GRPO": {42: 1.74, 98: 1.16, 5075: 1.73},
    },
}

# Return & success (env total reward from analyze_reward.py)
RETURN_DATA = {
    "PPO Misaligned":  {42: (425.0, 0.0), 98: (428.8, 0.0), 5075: (440.3, 0.0)},
    "GRPO Misaligned": {42: (211.5, 2.0), 98: (200.9, 0.0), 5075: (219.1, 2.0)},
    "PPO Sparse":  {42: (118.4, 0.0), 98: (218.5, 0.0), 5075: (213.8, 1.0)},
    "GRPO Sparse": {42: (192.0, 1.0), 98: (200.7, 0.0), 5075: (188.1, 3.0)},
}
BASELINE_RETURN = 244.0
BASELINE_SUCCESS = 5.0


# ── Helpers ──────────────────────────────────────────────────────────────────

def mean_std_from_dict(d: dict) -> tuple:
    """Compute mean and std from a {seed: value} dict."""
    vals = list(d.values())
    return np.mean(vals), np.std(vals)


def load_reward_decomp_multi() -> dict:
    """Load reward decomp from multiple seed CSVs, return per-model summary.

    Returns dict: model_name -> {seed: {"door_openness": ..., "passage": ..., "others": ...}}
    """
    models = ["Baseline", "PPO", "GRPO", "PPO-v3", "GRPO-v3"]
    csv_files = {
        42:   OUT_DIR / "reward_decomp.csv",
        98:   OUT_DIR / "reward_decomp_98.csv",
        5075: OUT_DIR / "reward_decomp_5075.csv",
    }

    summary = {m: {} for m in models}

    for seed, csv_path in csv_files.items():
        if not csv_path.exists():
            print(f"  [WARN] Missing: {csv_path} — skipping seed {seed}")
            continue
        df = pd.read_csv(csv_path)
        for m in models:
            sub = df[df["model"] == m]
            if len(sub) == 0:
                continue
            stand_w = sub["stand_weighted"].mean()
            hatch_w = (0.05 * sub["door_hatch_openness_reward"]).mean()
            hand_w  = (0.05 * sub["hand_hatch_proximity_reward"]).mean()
            summary[m][seed] = {
                "door_openness": 0.45 * sub["door_openness_reward"].mean(),
                "passage":       0.35 * sub["passage_reward"].mean(),
                "others": stand_w + hatch_w + hand_w,
                # per-episode fraction mean, consistent with the tables:
                # door% = mean(door_i / total_i), not mean(door_i)/mean(total_i)
                "door_frac": np.mean(
                    (0.45 * sub["door_openness_reward"] / sub["total_reward"]).values
                ),
            }
    return summary


# ── Figure 1: Reward Decomposition (multi-seed, mean + individual dots) ───────

def plot_reward_decomp(summary: dict):
    out = OUT_DIR / "fig_reward_decomp.png"
    # summary keys come from load_reward_decomp_multi (CSV model names)
    model_keys = ["Baseline", "PPO", "GRPO", "PPO-v3", "GRPO-v3"]
    labels = ["Baseline", "PPO (Misaligned)", "GRPO (Misaligned)", "PPO (Sparse)", "GRPO (Sparse)"]
    x = np.arange(len(model_keys))
    width = 0.55

    # Compute mean across seeds
    means = {}
    for m in model_keys:
        seeds_data = summary.get(m, {})
        if not seeds_data:
            means[m] = {"door_openness": 0, "passage": 0, "others": 0}
            continue
        means[m] = {
            "door_openness": np.mean([s["door_openness"] for s in seeds_data.values()]),
            "passage":       np.mean([s["passage"] for s in seeds_data.values()]),
            "others":        np.mean([s["others"] for s in seeds_data.values()]),
        }

    door_vals  = np.array([means[m]["door_openness"] for m in model_keys])
    passage_vals = np.array([means[m]["passage"] for m in model_keys])
    others_vals  = np.array([means[m]["others"] for m in model_keys])
    totals = door_vals + passage_vals + others_vals

    fig, ax = plt.subplots(figsize=(9, 5.5))

    ax.bar(x, others_vals, width, label="Other components\n(stand, hatch, hand prox)",
           color=COMP_COLORS["others"], edgecolor="#fcfcfb", linewidth=2)
    ax.bar(x, passage_vals, width, bottom=others_vals,
           label="Passage\n(true objective)",
           color=COMP_COLORS["passage"], edgecolor="#fcfcfb", linewidth=2)
    ax.bar(x, door_vals, width, bottom=others_vals + passage_vals,
           label="Door Openness",
           color=COMP_COLORS["door_openness"], edgecolor="#fcfcfb", linewidth=2)

    # Overlay individual seed totals as dots
    for i, m in enumerate(model_keys):
        seeds_data = summary.get(m, {})
        for seed, sd in seeds_data.items():
            seed_total = sd["door_openness"] + sd["passage"] + sd["others"]
            ax.scatter(i, seed_total, color="#0b0b0b", s=30, zorder=5,
                       edgecolors="#fcfcfb", linewidths=0.8, alpha=0.7)

    # Mean total labels
    for i, t in enumerate(totals):
        ax.text(i, t + 7, f"{t:.0f}", ha="center", va="bottom",
                fontsize=10, fontweight="bold", color="#0b0b0b")
    # Door % labels: per-episode fraction mean, consistent with the tables
    for i, m in enumerate(model_keys):
        seeds_data = summary.get(m, {})
        if not seeds_data:
            continue
        fracs = [s["door_frac"] for s in seeds_data.values()]
        pct = np.mean(fracs) * 100
        if pct > 20:  # PPO (Sparse) is ~36%; show it too
            y_pos = others_vals[i] + passage_vals[i] + door_vals[i] / 2
            ax.text(i, y_pos, f"{pct:.0f}%", ha="center", va="center",
                    fontsize=9, fontweight="bold", color="#fcfcfb")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylabel("Weighted Reward Contribution", fontsize=11)
    ax.set_title("Reward Decomposition (mean across 3 seeds, dots = individual seeds)",
                 fontweight="bold", loc="left")
    ax.set_ylim(0, totals.max() * 1.22)
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    fig.savefig(out)
    print(f"[DONE] → {out}")
    plt.close(fig)


# ── Figure 2: KL Divergence (multi-seed dots + mean bars) ─────────────────────

def plot_kl():
    out = OUT_DIR / "fig_kl_divergence.png"
    phases = list(KL_MULTISEED.keys())
    n_phases = len(phases)
    algos = ["PPO", "GRPO"]
    colors = [PALETTE["PPO"], PALETTE["GRPO"]]

    fig, ax = plt.subplots(figsize=(8, 5.5))

    # Compute means and collect individual values
    bar_positions = []
    bar_heights = []
    bar_colors = []
    bar_labels = []
    dot_data = []  # (x, y, color)

    for pi, phase in enumerate(phases):
        for ai, algo in enumerate(algos):
            vals = list(KL_MULTISEED[phase][algo].values())
            mu = np.mean(vals)
            x_pos = pi * 2.5 + ai * 0.8  # spread groups
            bar_positions.append(x_pos)
            bar_heights.append(mu)
            bar_colors.append(colors[ai])
            bar_labels.append(f"{algo}")
            # Individual seed dots
            for v in vals:
                dot_data.append((x_pos, v, colors[ai]))

    bars = ax.bar(bar_positions, bar_heights, 0.55, color=bar_colors,
                  edgecolor="#fcfcfb", linewidth=2, zorder=2)

    # Individual seed dots
    for x, y, c in dot_data:
        ax.scatter(x, y, color=c, s=50, edgecolors="#fcfcfb",
                   linewidths=1.2, zorder=5, alpha=0.9)

    # Value labels
    for bar, h in zip(bars, bar_heights):
        ax.text(bar.get_x() + bar.get_width() / 2, h + 1.2,
                f"{h:.1f}", ha="center", va="bottom", fontsize=9,
                fontweight="bold", color="#0b0b0b")

    # Phase labels
    for pi, phase in enumerate(phases):
        x_center = pi * 2.5 + 0.4
        ax.text(x_center, -3.5, phase, ha="center", fontsize=10,
                fontweight="bold", color="#52514e")

    # Custom legend
    legend_handles = [Patch(color=PALETTE["PPO"]), Patch(color=PALETTE["GRPO"])]
    ax.legend(legend_handles, ["PPO", "GRPO"], loc="upper left", fontsize=10)
    ax.set_xticks([])
    ax.set_ylabel("KL Divergence (nats)", fontsize=11)
    ax.set_title("KL(π_finetuned || π_baseline): Multi-Seed Comparison\n(dots = individual seeds, bars = mean)",
                 fontweight="bold", loc="left")
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.set_ylim(0, max(bar_heights) * 1.22)

    fig.savefig(out)
    print(f"[DONE] → {out}")
    plt.close(fig)


# ── Figure 3: KL vs Return scatter (NEW — the decoupling story) ───────────────

def plot_kl_vs_return():
    out = OUT_DIR / "fig_kl_vs_return.png"
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))
    fig.subplots_adjust(wspace=0.3)

    for ax, phase, kl_key, ret_ppo_key, ret_grpo_key in [
        (ax1, "Misaligned Reward", "Misaligned Reward", "PPO Misaligned", "GRPO Misaligned"),
        (ax2, "Sparse Reward", "Sparse Reward", "PPO Sparse", "GRPO Sparse"),
    ]:
        # PPO points
        for seed in SEEDS:
            kl = KL_MULTISEED[kl_key]["PPO"].get(seed)
            ret, succ = RETURN_DATA[ret_ppo_key].get(seed, (0, 0))
            if kl is not None:
                ax.scatter(kl, ret, color=PALETTE["PPO"], s=100, zorder=4,
                           edgecolors="#fcfcfb", linewidths=1.5, alpha=0.9)
                ax.annotate(f"s={seed}", (kl, ret), (kl + 0.8, ret + 2),
                            fontsize=7, color=PALETTE["PPO"], alpha=0.7)

        # GRPO points
        for seed in SEEDS:
            kl = KL_MULTISEED[kl_key]["GRPO"].get(seed)
            ret, succ = RETURN_DATA[ret_grpo_key].get(seed, (0, 0))
            if kl is not None:
                ax.scatter(kl, ret, color=PALETTE["GRPO"], s=100, zorder=4,
                           edgecolors="#fcfcfb", linewidths=1.5, alpha=0.9)
                ax.annotate(f"s={seed}", (kl, ret), (kl + 0.8, ret + 2),
                            fontsize=7, color=PALETTE["GRPO"], alpha=0.7)

        # Highlight PPO's KL-return covariation in the Misaligned panel
        if phase == "Misaligned Reward":
            ppo_pts = sorted(
                (KL_MULTISEED[kl_key]["PPO"][s], RETURN_DATA[ret_ppo_key][s][0])
                for s in SEEDS if s in KL_MULTISEED[kl_key]["PPO"]
            )
            if len(ppo_pts) >= 2:
                ax.plot([p[0] for p in ppo_pts], [p[1] for p in ppo_pts],
                        color=PALETTE["PPO"], linestyle=":", linewidth=1.3,
                        alpha=0.55, zorder=3)

        # Baseline reference
        ax.axhline(y=BASELINE_RETURN, color="#898781", linestyle="--",
                   linewidth=1.0, alpha=0.7, label=f"Baseline ({BASELINE_RETURN:.0f})")

        ax.set_title(phase, fontweight="bold", loc="left", fontsize=11)
        ax.set_xlabel("KL Divergence (nats)", fontsize=10)
        if ax == ax1:
            ax.set_ylabel("Mean Return", fontsize=10)

        # Legend
        legend_handles = [
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=PALETTE["PPO"],
                        markersize=8, label='PPO'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=PALETTE["GRPO"],
                        markersize=8, label='GRPO'),
            plt.Line2D([0], [0], color='#898781', linestyle='--', label='Baseline'),
        ]
        ax.legend(handles=legend_handles, loc="best", fontsize=8)
        ax.grid(linestyle="--", alpha=0.4)

    fig.suptitle("KL–Return Decoupling: GRPO Return is Stable Regardless of KL",
                 fontweight="bold", fontsize=13, y=1.02)
    fig.savefig(out)
    print(f"[DONE] → {out}")
    plt.close(fig)


# ── Figure 4: Parameter Distance (multi-seed) ─────────────────────────────────

def plot_param_distance():
    out = OUT_DIR / "fig_param_distance.png"
    phases = list(PARAM_MULTISEED.keys())
    algos = ["PPO", "GRPO"]
    n_groups = len(phases)

    fig, ax = plt.subplots(figsize=(7.5, 5.5))

    for pi, phase in enumerate(phases):
        for ai, algo in enumerate(algos):
            vals = list(PARAM_MULTISEED[phase][algo].values())
            mu = np.mean(vals)
            x_pos = pi * 2.5 + ai * 0.8
            color = PALETTE[algo]
            ax.bar(x_pos, mu, 0.55, color=color, edgecolor="#fcfcfb",
                   linewidth=2, zorder=2)
            for v in vals:
                ax.scatter(x_pos, v, color=color, s=50, edgecolors="#fcfcfb",
                           linewidths=1.2, zorder=5, alpha=0.9)
            ax.text(x_pos, mu + 0.12, f"{mu:.2f}", ha="center", va="bottom",
                    fontsize=9, fontweight="bold", color="#0b0b0b")

    for pi, phase in enumerate(phases):
        x_center = pi * 2.5 + 0.4
        ax.text(x_center, -0.25, phase, ha="center", fontsize=10,
                fontweight="bold", color="#52514e")

    legend_handles = [Patch(color=PALETTE["PPO"]), Patch(color=PALETTE["GRPO"])]
    ax.legend(legend_handles, ["PPO", "GRPO"], loc="upper left", fontsize=10)
    ax.set_xticks([])
    ax.set_ylabel("Total L2 Distance from Baseline", fontsize=11)
    ax.set_title("Parameter-Space Distance: ||θ_finetune − θ_baseline||₂\n(dots = individual seeds, bars = mean)",
                 fontweight="bold", loc="left")
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.set_ylim(0, max(PARAM_MULTISEED["Misaligned Reward"]["PPO"].values()) * 1.22)

    fig.savefig(out)
    print(f"[DONE] → {out}")
    plt.close(fig)


# ── Figure 5: Action Mean Difference (multi-seed) ─────────────────────────────

def plot_action_diff():
    out = OUT_DIR / "fig_action_diff.png"
    phases = list(ACTION_DIFF_MULTISEED.keys())
    algos = ["PPO", "GRPO"]

    fig, ax = plt.subplots(figsize=(7.5, 5.5))

    for pi, phase in enumerate(phases):
        for ai, algo in enumerate(algos):
            vals = list(ACTION_DIFF_MULTISEED[phase][algo].values())
            mu = np.mean(vals)
            x_pos = pi * 2.5 + ai * 0.8
            color = PALETTE[algo]
            ax.bar(x_pos, mu, 0.55, color=color, edgecolor="#fcfcfb",
                   linewidth=2, zorder=2)
            for v in vals:
                ax.scatter(x_pos, v, color=color, s=50, edgecolors="#fcfcfb",
                           linewidths=1.2, zorder=5, alpha=0.9)
            ax.text(x_pos, mu + 0.08, f"{mu:.2f}", ha="center", va="bottom",
                    fontsize=9, fontweight="bold", color="#0b0b0b")

    for pi, phase in enumerate(phases):
        x_center = pi * 2.5 + 0.4
        ax.text(x_center, -0.18, phase, ha="center", fontsize=10,
                fontweight="bold", color="#52514e")

    legend_handles = [Patch(color=PALETTE["PPO"]), Patch(color=PALETTE["GRPO"])]
    ax.legend(legend_handles, ["PPO", "GRPO"], loc="upper left", fontsize=10)
    ax.set_xticks([])
    ax.set_ylabel("Mean ||Δμ||₂", fontsize=11)
    ax.set_title("Deterministic Action Shift from Baseline\n(dots = individual seeds, bars = mean)",
                 fontweight="bold", loc="left")
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.set_ylim(0, max(ACTION_DIFF_MULTISEED["Sparse Reward"]["PPO"].values()) * 1.22)

    fig.savefig(out)
    print(f"[DONE] → {out}")
    plt.close(fig)


# ── Figure 6: Log-Std Shift (multi-seed) ──────────────────────────────────────

def plot_logstd_shift():
    out = OUT_DIR / "fig_logstd_shift.png"
    phases = list(LOGSTD_MULTISEED.keys())
    algos = ["PPO", "GRPO"]

    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    ax.axhline(y=0, color="#0b0b0b", linewidth=0.8, zorder=1)

    for pi, phase in enumerate(phases):
        for ai, algo in enumerate(algos):
            vals = list(LOGSTD_MULTISEED[phase][algo].values())
            mu = np.mean(vals)
            x_pos = pi * 2.5 + ai * 0.8
            color = PALETTE[algo]
            ax.bar(x_pos, mu, 0.55, color=color, edgecolor="#fcfcfb",
                   linewidth=2, zorder=2)
            for v in vals:
                ax.scatter(x_pos, v, color=color, s=50, edgecolors="#fcfcfb",
                           linewidths=1.2, zorder=5, alpha=0.9)
            va = "bottom" if mu >= 0 else "top"
            y_off = 0.003 if mu >= 0 else -0.003
            ax.text(x_pos, mu + y_off, f"{mu:+.4f}", ha="center", va=va,
                    fontsize=8, fontweight="bold")

    for pi, phase in enumerate(phases):
        x_center = pi * 2.5 + 0.4
        ax.text(x_center, -0.088, phase, ha="center", fontsize=10,
                fontweight="bold", color="#52514e")

    legend_handles = [Patch(color=PALETTE["PPO"]), Patch(color=PALETTE["GRPO"])]
    ax.legend(legend_handles, ["PPO", "GRPO"], loc="lower left", fontsize=10)
    ax.set_xticks([])
    ax.set_ylabel("Mean Δ log_σ from Baseline", fontsize=11)
    ax.set_title("Policy Entropy Shift: Δ log_σ from Baseline\n(dots = individual seeds, bars = mean)",
                 fontweight="bold", loc="left")
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    fig.savefig(out)
    print(f"[DONE] → {out}")
    plt.close(fig)


# ── Figure 7: Return vs Success (multi-seed) ──────────────────────────────────

def plot_return_vs_success():
    out = OUT_DIR / "fig_return_vs_success.png"
    fig, ax = plt.subplots(figsize=(8, 5.5))

    # Baseline
    ax.scatter(BASELINE_RETURN, BASELINE_SUCCESS, c="#898781", s=140,
               edgecolors="#fcfcfb", linewidths=2, zorder=3, marker="s")
    ax.annotate("Baseline", (BASELINE_RETURN, BASELINE_SUCCESS),
                (BASELINE_RETURN - 30, BASELINE_SUCCESS + 0.3),
                fontsize=9, ha="center", color="#898781", zorder=10,
                arrowprops=dict(arrowstyle="-", color="#e1e0d9", lw=0.8))

    # PPO Misaligned
    for seed, (ret, succ) in RETURN_DATA["PPO Misaligned"].items():
        ax.scatter(ret, succ, color=PALETTE["PPO"], s=90, zorder=4,
                   edgecolors="#fcfcfb", linewidths=1.5, alpha=0.8)
    # GRPO Misaligned
    for seed, (ret, succ) in RETURN_DATA["GRPO Misaligned"].items():
        ax.scatter(ret, succ, color=PALETTE["GRPO"], s=90, zorder=4,
                   edgecolors="#fcfcfb", linewidths=1.5, alpha=0.8)
    # PPO Sparse
    for seed, (ret, succ) in RETURN_DATA["PPO Sparse"].items():
        ax.scatter(ret, succ, color=PALETTE["PPO-v3"], s=90, zorder=4,
                   edgecolors="#fcfcfb", linewidths=1.5, alpha=0.8, marker="^")
    # GRPO Sparse
    for seed, (ret, succ) in RETURN_DATA["GRPO Sparse"].items():
        ax.scatter(ret, succ, color=PALETTE["GRPO-v3"], s=90, zorder=4,
                   edgecolors="#fcfcfb", linewidths=1.5, alpha=0.8, marker="^")

    ax.axhline(y=BASELINE_SUCCESS, color="#898781", linestyle="--",
               linewidth=0.8, alpha=0.5)

    # Custom legend
    legend_handles = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=PALETTE["PPO"],
                    markersize=8, label='PPO (Misaligned)'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=PALETTE["GRPO"],
                    markersize=8, label='GRPO (Misaligned)'),
        plt.Line2D([0], [0], marker='^', color='w', markerfacecolor=PALETTE["PPO-v3"],
                    markersize=8, label='PPO (Sparse)'),
        plt.Line2D([0], [0], marker='^', color='w', markerfacecolor=PALETTE["GRPO-v3"],
                    markersize=8, label='GRPO (Sparse)'),
        plt.Line2D([0], [0], marker='s', color='w', markerfacecolor="#898781",
                    markersize=8, label='Baseline'),
    ]
    ax.legend(handles=legend_handles, loc="upper right", fontsize=8)

    ax.set_xlabel("Mean Return", fontsize=11)
    ax.set_ylabel("Success Rate (%)", fontsize=11)
    ax.set_ylim(-0.5, 6.5)
    ax.set_title("Return vs Success: Multi-Seed (3 seeds)\n(circles = Misaligned, triangles = Sparse)",
                 fontweight="bold", loc="left")
    ax.grid(linestyle="--", alpha=0.4)

    fig.savefig(out)
    print(f"[DONE] → {out}")
    plt.close(fig)


# ── Figure 8: Combined Overview (4 panels) ────────────────────────────────────

def plot_combined_overview(summary: dict):
    out = OUT_DIR / "fig_overview.png"
    fig = plt.figure(figsize=(15, 10))

    # ─ Panel A: Reward Decomp ────────────────────────────────────────────
    ax_a = fig.add_subplot(2, 2, 1)
    models = ["Baseline", "PPO", "GRPO", "PPO-v3", "GRPO-v3"]
    x = np.arange(len(models))
    width = 0.55

    means = {}
    for m in models:
        seeds_data = summary.get(m, {})
        if not seeds_data:
            means[m] = {"door_openness": 0, "passage": 0, "others": 0}
            continue
        means[m] = {
            "door_openness": np.mean([s["door_openness"] for s in seeds_data.values()]),
            "passage":       np.mean([s["passage"] for s in seeds_data.values()]),
            "others":        np.mean([s["others"] for s in seeds_data.values()]),
        }

    door_vals  = np.array([means[m]["door_openness"] for m in models])
    passage_vals = np.array([means[m]["passage"] for m in models])
    others_vals  = np.array([means[m]["others"] for m in models])
    totals = door_vals + passage_vals + others_vals

    ax_a.bar(x, others_vals, width, label="Other components",
             color=COMP_COLORS["others"], edgecolor="#fcfcfb", linewidth=1.5)
    ax_a.bar(x, passage_vals, width, bottom=others_vals,
             label="Passage (true objective)",
             color=COMP_COLORS["passage"], edgecolor="#fcfcfb", linewidth=1.5)
    ax_a.bar(x, door_vals, width, bottom=others_vals + passage_vals,
             label="Door Openness",
             color=COMP_COLORS["door_openness"], edgecolor="#fcfcfb", linewidth=1.5)
    for i, m in enumerate(models):
        seeds_data = summary.get(m, {})
        for seed, sd in seeds_data.items():
            seed_total = sd["door_openness"] + sd["passage"] + sd["others"]
            ax_a.scatter(i, seed_total, color="#0b0b0b", s=20, zorder=5,
                         edgecolors="#fcfcfb", linewidths=0.5, alpha=0.7)
    for i, t in enumerate(totals):
        ax_a.text(i, t + 4, f"{t:.0f}", ha="center", va="bottom",
                  fontsize=9, fontweight="bold")
    ax_a.set_xticks(x)
    ax_a.set_xticklabels(models, fontsize=9)
    ax_a.set_ylabel("Weighted Reward")
    ax_a.set_title("A  Reward Decomposition (mean, 3 seeds)", fontweight="bold",
                   loc="left", fontsize=11)
    ax_a.legend(loc="upper right", fontsize=7)
    ax_a.grid(axis="y", linestyle="--", alpha=0.4)

    # ─ Panel B: KL Divergence ────────────────────────────────────────────
    ax_b = fig.add_subplot(2, 2, 2)
    phases = list(KL_MULTISEED.keys())
    algos = ["PPO", "GRPO"]
    colors_k = [PALETTE["PPO"], PALETTE["GRPO"]]
    bar_positions_b = []
    bar_heights_b = []
    bar_colors_b = []

    for pi, phase in enumerate(phases):
        for ai, algo in enumerate(algos):
            vals = list(KL_MULTISEED[phase][algo].values())
            mu = np.mean(vals)
            x_pos = pi * 2.5 + ai * 0.8
            bar_positions_b.append(x_pos)
            bar_heights_b.append(mu)
            bar_colors_b.append(colors_k[ai])
            for v in vals:
                ax_b.scatter(x_pos, v, color=colors_k[ai], s=35,
                             edgecolors="#fcfcfb", linewidths=0.8, zorder=5, alpha=0.9)

    ax_b.bar(bar_positions_b, bar_heights_b, 0.55, color=bar_colors_b,
             edgecolor="#fcfcfb", linewidth=1.5, zorder=2)
    for pi, phase in enumerate(phases):
        ax_b.text(pi * 2.5 + 0.4, -3.5, phase, ha="center", fontsize=9,
                  fontweight="bold", color="#52514e")

    legend_handles_b = [Patch(color=PALETTE["PPO"]), Patch(color=PALETTE["GRPO"])]
    ax_b.legend(legend_handles_b, ["PPO", "GRPO"], loc="upper left", fontsize=8)
    ax_b.set_xticks([])
    ax_b.set_ylabel("KL (nats)")
    ax_b.set_title("B  Policy KL from Baseline (3 seeds)", fontweight="bold",
                   loc="left", fontsize=11)
    ax_b.grid(axis="y", linestyle="--", alpha=0.4)

    # ─ Panel C: KL vs Return ─────────────────────────────────────────────
    ax_c = fig.add_subplot(2, 2, 3)
    for seed in SEEDS:
        for kl_key, ret_key, color, marker in [
            ("Misaligned Reward", "PPO Misaligned", PALETTE["PPO"], "o"),
            ("Misaligned Reward", "GRPO Misaligned", PALETTE["GRPO"], "o"),
            ("Sparse Reward", "PPO Sparse", PALETTE["PPO-v3"], "^"),
            ("Sparse Reward", "GRPO Sparse", PALETTE["GRPO-v3"], "^"),
        ]:
            kl = KL_MULTISEED[kl_key]["PPO" if "PPO" in ret_key else "GRPO"].get(seed)
            ret_data = RETURN_DATA[ret_key].get(seed)
            if kl is not None and ret_data is not None:
                ret, _ = ret_data
                ax_c.scatter(kl, ret, color=color, s=65, marker=marker,
                             edgecolors="#fcfcfb", linewidths=1.0, zorder=4, alpha=0.85)

    ax_c.axhline(y=BASELINE_RETURN, color="#898781", linestyle="--",
                 linewidth=1.0, alpha=0.7, label=f"Baseline ({BASELINE_RETURN:.0f})")
    ax_c.set_xlabel("KL Divergence (nats)")
    ax_c.set_ylabel("Mean Return")
    ax_c.set_title("C  KL–Return Decoupling", fontweight="bold", loc="left", fontsize=11)
    ax_c.legend(fontsize=7)
    ax_c.grid(linestyle="--", alpha=0.4)

    # ─ Panel D: Log-Std Shift ────────────────────────────────────────────
    ax_d = fig.add_subplot(2, 2, 4)
    phases_d = list(LOGSTD_MULTISEED.keys())
    bar_positions_d = []
    bar_heights_d = []
    bar_colors_d = []

    for pi, phase in enumerate(phases_d):
        for ai, algo in enumerate(algos):
            vals = list(LOGSTD_MULTISEED[phase][algo].values())
            mu = np.mean(vals)
            x_pos = pi * 2.5 + ai * 0.8
            bar_positions_d.append(x_pos)
            bar_heights_d.append(mu)
            bar_colors_d.append(PALETTE[algo])
            for v in vals:
                ax_d.scatter(x_pos, v, color=PALETTE[algo], s=35,
                             edgecolors="#fcfcfb", linewidths=0.8, zorder=5, alpha=0.9)

    ax_d.bar(bar_positions_d, bar_heights_d, 0.55, color=bar_colors_d,
             edgecolor="#fcfcfb", linewidth=1.5, zorder=2)
    ax_d.axhline(y=0, color="#0b0b0b", linewidth=0.8)
    for pi, phase in enumerate(phases_d):
        ax_d.text(pi * 2.5 + 0.4, -0.088, phase, ha="center", fontsize=9,
                  fontweight="bold", color="#52514e")

    legend_handles_d = [Patch(color=PALETTE["PPO"]), Patch(color=PALETTE["GRPO"])]
    ax_d.legend(legend_handles_d, ["PPO", "GRPO"], loc="lower left", fontsize=8)
    ax_d.set_xticks([])
    ax_d.set_ylabel("Mean Δ log_σ from Baseline")
    ax_d.set_title("D  Policy Entropy Shift (3 seeds)", fontweight="bold",
                   loc="left", fontsize=11)
    ax_d.grid(axis="y", linestyle="--", alpha=0.4)

    fig.tight_layout(pad=2.5)
    fig.savefig(out)
    print(f"[DONE] → {out}")
    plt.close(fig)


# ── Entry point ──────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("Multi-Seed Visualization: Analysis Figures")
    print("=" * 60)

    # Load reward decomp from multiple CSVs
    try:
        summary = load_reward_decomp_multi()
        n_models = len(summary)
        n_seeds = max(len(v) for v in summary.values())
        print(f"[DATA] Reward decomp: {n_models} models, up to {n_seeds} seeds")
    except Exception as e:
        print(f"[WARN] Reward decomp loading failed: {e}")
        summary = {}

    if summary:
        plot_reward_decomp(summary)
    plot_kl()
    plot_kl_vs_return()
    plot_param_distance()
    plot_action_diff()
    plot_logstd_shift()
    plot_return_vs_success()
    if summary:
        plot_combined_overview(summary)

    print(f"\nAll figures saved to: {OUT_DIR}/")


if __name__ == "__main__":
    main()