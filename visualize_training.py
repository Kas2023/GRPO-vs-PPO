#!/usr/bin/env python3
"""
Training curves: PPO vs GRPO on the Door task — multi-seed version.

Two panels:
  A — Phase 1 (dense reward): eval mean return over timesteps.
      PPO return skyrockets (reward hacking); GRPO stays flat.
  B — Phase 2 (sparse binary reward): eval success rate.

Each line = mean across seeds, with ±1 std shaded band.
Individual seed curves shown as faint lines.

Data source: logs/h1hand-door-v0/{exp}/eval/evaluations.npy
Output:      results/fig_training_curves.png
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from pathlib import Path

# ── Palette ──────────────────────────────────────────────────────────────────
PALETTE = {
    "PPO":      "#eb6834",
    "GRPO":     "#2a78d6",
    "PPO-v3":   "#eda100",
    "GRPO-v3":  "#1baf7a",
}

# Each experiment key maps to a list of seed dirs
EXP_CONFIG = {
    "PPO":      {"dirs": ["ft_ppo_v2", "ft_ppo_v2_98", "ft_ppo_v2_5075"],
                 "label": "PPO (Misaligned)", "phase": "misaligned"},
    "GRPO":     {"dirs": ["ft_grpo_v2", "ft_grpo_v2_98", "ft_grpo_v2_5075"],
                 "label": "GRPO (Misaligned)", "phase": "misaligned"},
    "PPO-v3":   {"dirs": ["ft_ppo_v3", "ft_ppo_v3_98", "ft_ppo_v3_5075"],
                 "label": "PPO (Sparse)", "phase": "sparse"},
    "GRPO-v3":  {"dirs": ["ft_grpo_v3", "ft_grpo_v3_98", "ft_grpo_v3_5075"],
                 "label": "GRPO (Sparse)", "phase": "sparse"},
}

LOG_ROOT = Path(__file__).resolve().parent / "logs" / "h1hand-door-v0"
OUT_DIR  = Path(__file__).resolve().parent / "results"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Global style ─────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica", "sans-serif"],
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.8,
    "axes.edgecolor": "#c3c2b7",
    "xtick.color": "#898781",
    "ytick.color": "#898781",
    "grid.color": "#e1e0d9",
    "grid.linewidth": 0.5,
    "legend.frameon": True,
    "legend.framealpha": 0.9,
    "legend.edgecolor": "#e1e0d9",
    "legend.fontsize": 9,
    "figure.facecolor": "#fcfcfb",
    "axes.facecolor": "#fcfcfb",
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
    "savefig.facecolor": "#fcfcfb",
})

# ── Helpers ──────────────────────────────────────────────────────────────────

def load_eval_data(exp_dir_name: str) -> dict | None:
    npy_path = LOG_ROOT / exp_dir_name / "eval" / "evaluations.npy"
    if not npy_path.exists():
        return None
    data = np.load(npy_path, allow_pickle=True).item()
    return {
        "timesteps": np.array(data["timesteps"]) / 1e6,  # in millions
        "mean": np.array([r[0] for r in data["results"]]),
        "std":  np.array([r[1] for r in data["results"]]),
    }


def compute_band(seeds_data: list[dict]) -> dict:
    """Given a list of per-seed eval dicts, interpolate to a common
    timestep grid and compute mean ± std across seeds."""
    # Find the union of all timestep points, sorted
    all_ts = np.unique(np.concatenate([d["timesteps"] for d in seeds_data]))
    # Use a common fine grid
    t_min = max(d["timesteps"][0] for d in seeds_data)
    t_max = min(d["timesteps"][-1] for d in seeds_data)
    if t_min >= t_max:
        return {"timesteps": None, "mean": None, "std": None, "seeds": seeds_data}
    common_t = np.linspace(t_min, t_max, 200)

    interp_vals = []
    for d in seeds_data:
        interp = np.interp(common_t, d["timesteps"], d["mean"])
        interp_vals.append(interp)

    interp_vals = np.array(interp_vals)
    return {
        "timesteps": common_t,
        "mean": interp_vals.mean(axis=0),
        "std": interp_vals.std(axis=0),
        "seeds": seeds_data,
    }


def _fmt_M(v, _):
    if v >= 1:
        return f"{v:.0f}M"
    return f"{v*1000:.0f}K"


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    out = OUT_DIR / "fig_training_curves.png"

    # Load all seed data
    results = {}
    for label, cfg in EXP_CONFIG.items():
        seeds_data = []
        for d in cfg["dirs"]:
            data = load_eval_data(d)
            if data is not None:
                seeds_data.append(data)
                print(f"[LOAD] {label:8s}  seed dir {d:20s}  "
                      f"({len(data['timesteps'])} pts, "
                      f"t=[{data['timesteps'][0]:.1f}M, {data['timesteps'][-1]:.1f}M])")
        if seeds_data:
            results[label] = compute_band(seeds_data)
            print(f"  → {label} aggregated: {len(seeds_data)} seeds, "
                  f"common t=[{results[label]['timesteps'][0]:.1f}M, "
                  f"{results[label]['timesteps'][-1]:.1f}M]")
        else:
            print(f"[SKIP] {label:8s} → no seeds found")

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7.5))
    fig.subplots_adjust(hspace=0.38)

    # ── Panel A: Phase 1 — eval mean return ───────────────────────────────
    for label in ["PPO", "GRPO"]:
        if label not in results or results[label]["timesteps"] is None:
            continue
        r = results[label]
        c = PALETTE[label]

        # Faint individual seed lines
        for sd in r["seeds"]:
            ax1.plot(sd["timesteps"], sd["mean"], color=c, linewidth=0.4, alpha=0.3)

        # Mean line
        ax1.plot(r["timesteps"], r["mean"], color=c, linewidth=2.2,
                 label=EXP_CONFIG[label]["label"],
                 marker="o", markersize=5, markevery=max(1, len(r["timesteps"]) // 10),
                 markerfacecolor=c, markeredgecolor="#fcfcfb", markeredgewidth=1.2)
        # ±1 std band
        ax1.fill_between(r["timesteps"],
                         np.maximum(0, r["mean"] - r["std"]),
                         r["mean"] + r["std"],
                         color=c, alpha=0.12, linewidth=0)

    ax1.axhline(y=243.9, color="#898781", linestyle="--", linewidth=1.0,
                alpha=0.8, label="Baseline (244)")

    ax1.set_title("A  Misaligned Reward: Eval Mean Return (3 seeds)",
                  fontweight="bold", loc="left")
    ax1.set_ylabel("Mean Return")
    ax1.legend(loc="upper left", fontsize=9)
    ax1.xaxis.set_major_formatter(mticker.FuncFormatter(_fmt_M))
    ax1.grid(True, linestyle="--", alpha=0.5)

    # ── Panel B: Phase 2 — eval success rate ──────────────────────────────
    for label in ["PPO-v3", "GRPO-v3"]:
        if label not in results or results[label]["timesteps"] is None:
            continue
        r = results[label]
        c = PALETTE[label]

        # Faint individual seed lines
        for sd in r["seeds"]:
            ax2.plot(sd["timesteps"], sd["mean"], color=c, linewidth=0.4, alpha=0.3)

        # Mean line
        ax2.plot(r["timesteps"], r["mean"], color=c, linewidth=2.2,
                 label=EXP_CONFIG[label]["label"],
                 marker="o", markersize=5, markevery=max(1, len(r["timesteps"]) // 10),
                 markerfacecolor=c, markeredgecolor="#fcfcfb", markeredgewidth=1.2)
        # ±1 std band
        ax2.fill_between(r["timesteps"],
                         np.maximum(0, r["mean"] - r["std"]),
                         np.minimum(1, r["mean"] + r["std"]),
                         color=c, alpha=0.12, linewidth=0)

    ax2.set_title("B  Sparse Reward: Eval Success Rate (3 seeds)",
                  fontweight="bold", loc="left")
    ax2.set_xlabel("Timesteps (millions)")
    ax2.set_ylabel("Success Rate")
    ax2.set_ylim(-0.02, 0.10)
    ax2.legend(loc="upper right", fontsize=9)
    ax2.xaxis.set_major_formatter(mticker.FuncFormatter(_fmt_M))
    ax2.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))
    ax2.grid(True, linestyle="--", alpha=0.5)

    fig.savefig(out)
    print(f"\n[DONE] → {out}")
    plt.close(fig)


if __name__ == "__main__":
    main()