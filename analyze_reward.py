"""
Reward decomposition analysis for Door task fine-tuning experiments.

Loads baseline, PPO-finetuned, and GRPO-finetuned models, runs evaluation
rollouts collecting per-step reward components, and produces a comparison
table showing which reward terms each model is (over-)optimizing.

The Door task reward (from humanoid_bench/envs/door.py) is:
    total = 0.1 * stand_reward * small_control
          + 0.45 * door_openness_reward
          + 0.05 * door_hatch_openness_reward
          + 0.05 * hand_hatch_proximity_reward
          + 0.35 * passage_reward

This script collects each component from the per-step info dict and reports:
  - Per-episode accumulated components (mean ± std across episodes)
  - Per-episode component breakdown as fraction of total reward
  - Summary comparison table across models

Usage:
    # Phase 1 (dense reward) — the most informative for reward decomposition
    conda activate humanoidbench
    python analyze_reward.py \
        --baseline_exp curriculum-v7.1 \
        --ppo_exp ft_ppo_v2 --ppo_model best_model \
        --grpo_exp ft_grpo_v2 --grpo_model best_model \
        --seeds 1024 2024 777 88 13 \
        --n_episodes 50

    # Phase 2 (sparse oracle) — reward decomp less informative but still useful
    python analyze_reward.py \
        --baseline_exp curriculum-v7.1 \
        --ppo_exp ft_ppo_v3 --ppo_model best_by_reward \
        --grpo_exp ft_grpo_v3 --grpo_model best_by_reward \
        --phase2 \
        --seeds 1024 2024 777 88 13 \
        --n_episodes 50

    # All four models at once + CSV export
python analyze_reward.py \
    --baseline_exp curriculum-v7.1 \
    --ppo_exp ft_ppo_v2_98 --ppo_model best_by_reward \
    --grpo_exp ft_grpokl_98 --grpo_model best_by_reward \
    --ppo2_exp ft_ppo_v3_98 --ppo2_model best_by_reward \
    --grpo2_exp ft_grpokl_2_98 --grpo2_model best_by_reward \
    --phase2 --csv results/reward_decomp_kl_98.csv
"""

import argparse
import os
import sys
import csv
from collections import defaultdict

import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

import humanoid_bench
from utils.grpo import GRPO


# ────────────────────────────────────────────────────────────────────────────
# Reward component names and their weights in the Door task
# ────────────────────────────────────────────────────────────────────────────

REWARD_COMPONENTS = [
    "stand_reward",
    "small_control",
    "door_openness_reward",
    "door_hatch_openness_reward",
    "hand_hatch_proximity_reward",
    "passage_reward",
]

COMPONENT_WEIGHTS = {
    "stand_reward": 0.1,              # multiplied by small_control in env
    "small_control": 0.1,             # factor — see note below
    "door_openness_reward": 0.45,
    "door_hatch_openness_reward": 0.05,
    "hand_hatch_proximity_reward": 0.05,
    "passage_reward": 0.35,
}

# Additive terms of the env reward. small_control is NOT here: it is only a
# multiplicative factor inside the stand term (0.1 * stand_reward * small_control),
# already accounted for by the "stand_reward" row via ep["stand_weighted"].
# Listing it as its own weighted contribution would double-count it and push
# the fraction rows past 100%.
WEIGHTED_TERMS = [
    "stand_reward",
    "door_openness_reward",
    "door_hatch_openness_reward",
    "hand_hatch_proximity_reward",
    "passage_reward",
]

# Note: In the env, stand_reward is weighted as 0.1 * stand_reward * small_control,
# not 0.1 * stand_reward + 0.1 * small_control. We report the raw components
# and compute the weighted total ourselves for verification.

# ────────────────────────────────────────────────────────────────────────────
# Minimal wrapper — tracks success + collects reward components from info dict
# ────────────────────────────────────────────────────────────────────────────

class RewardDecompWrapper(gym.Wrapper):
    """Collects per-step reward components from the env info dict.

    The underlying HumanoidBench Task.step() puts each reward component into
    the info dict (see tasks.py line 68). This wrapper accumulates them across
    a full episode and stores accumulated values in its own info dict at the
    terminal step.

    Success is detected via MuJoCo state (robot_x > 0.8), NOT from the info
    dict — the raw Door task does NOT include ``robot_x`` or ``success`` in
    its per-step info (only the six reward components are present).
    """

    def __init__(self, env):
        super().__init__(env)
        self._episode_success = False
        self._prev_action = None
        self._episode_action_sqdiff = 0.0
        self._component_accum = {c: 0.0 for c in REWARD_COMPONENTS}
        # stand term is 0.1 * stand_reward_t * small_control_t summed PER STEP —
        # it cannot be reconstructed from the per-episode sums (Σ(a·b) ≠ Σa·Σb).
        self._stand_weighted_accum = 0.0
        self._total_reward = 0.0
        self._n_steps = 0

    def _get_robot_x(self) -> float:
        """Read the robot's x-coordinate directly from MuJoCo state."""
        return float(
            self.env.unwrapped.named.data.site_xpos["imu", "x"]
        )

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._episode_success = False
        self._prev_action = None
        self._episode_action_sqdiff = 0.0
        self._component_accum = {c: 0.0 for c in REWARD_COMPONENTS}
        self._stand_weighted_accum = 0.0
        self._total_reward = 0.0
        self._n_steps = 0
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        self._n_steps += 1

        # Collect reward components from info dict
        for c in REWARD_COMPONENTS:
            if c in info:
                self._component_accum[c] += float(info[c])
        # stand term must be accumulated as the per-step PRODUCT
        self._stand_weighted_accum += (
            0.1 * float(info.get("stand_reward", 0.0))
            * float(info.get("small_control", 0.0))
        )
        self._total_reward += float(reward)

        # Success: robot torso past door frame (same criterion as EvalMetricsWrapper)
        if self._get_robot_x() > 0.8:
            self._episode_success = True

        # Action smoothness
        if self._prev_action is not None:
            self._episode_action_sqdiff += float(
                np.sum(np.square(action - self._prev_action))
            )
        self._prev_action = action.copy()

        if terminated or truncated:
            info["_reward_components"] = dict(self._component_accum)
            info["_stand_weighted"] = self._stand_weighted_accum
            info["_total_reward_accum"] = self._total_reward
            info["_n_steps"] = self._n_steps
            if self._episode_success:
                info["success"] = 1
            info["action_smoothness"] = self._episode_action_sqdiff

        return obs, reward, terminated, truncated, info


# ────────────────────────────────────────────────────────────────────────────
# Environment factory
# ────────────────────────────────────────────────────────────────────────────

def make_env(env_name: str = "h1hand-door-v0"):
    def _init():
        env = gym.make(env_name)
        # gym.make already applies TimeLimit(max_episode_steps=1000) at
        # registration (see humanoid_bench/env.py). Do NOT add a second
        # TimeLimit — a nested one shifts the truncation step and makes
        # episode returns drift by a step vs eval.py's wrapper stack.
        env = RewardDecompWrapper(env)
        return env
    return _init


# ────────────────────────────────────────────────────────────────────────────
# Single-model evaluation
# ────────────────────────────────────────────────────────────────────────────

def eval_model(
    model,
    env_name: str,
    vecnormalize_path: str | None,
    n_episodes: int = 20,
    base_seed: int = 0,
):
    """Run evaluation rollouts and collect per-episode reward components.

    Returns a dict with:
        - per_episode: list of per-episode dicts with accumulated components
        - aggregate: mean ± std per component across all episodes
        - success_rate: fraction of successful episodes
    """

    env = DummyVecEnv([make_env(env_name)])

    # Load VecNormalize onto the RAW DummyVecEnv (single wrapper) — same as
    # eval.py. Wrapping a fresh VecNormalize first and then loading on top of
    # it double-normalizes: the inner fresh layer clips raw obs to ±clip_obs
    # (in raw units) BEFORE the real stats are applied, corrupting any obs
    # component whose magnitude exceeds clip_obs (e.g. joint velocities).
    if vecnormalize_path and os.path.exists(vecnormalize_path):
        env = VecNormalize.load(vecnormalize_path, env)
        env.training = False
        env.norm_reward = False
    else:
        env = VecNormalize(env, norm_obs=True, norm_reward=False,
                           clip_obs=10.0, training=False)

    model.set_env(env)

    per_episode = []
    success_count = 0

    for ep in range(n_episodes):
        env.venv.envs[0].reset(seed=base_seed + ep)
        obs = env.reset()
        done = False

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, dones, info = env.step(action)
            done = bool(dones[0])

        env_info = info[0]
        components = env_info.get("_reward_components", {})
        total_reward = env_info.get("_total_reward_accum", 0.0)
        ep_success = env_info.get("success", 0) == 1

        if ep_success:
            success_count += 1

        per_episode.append({
            "components": components,
            "stand_weighted": env_info.get("_stand_weighted", 0.0),
            "total_reward": total_reward,
            "success": ep_success,
            "n_steps": env_info.get("_n_steps", 0),
        })

    env.close()

    # Aggregate
    agg = {}
    for c in REWARD_COMPONENTS:
        vals = [ep["components"].get(c, 0.0) for ep in per_episode]
        agg[c] = {
            "mean": float(np.mean(vals)),
            "std": float(np.std(vals)),
            "min": float(np.min(vals)),
            "max": float(np.max(vals)),
        }

    total_returns = [ep["total_reward"] for ep in per_episode]
    agg["total_reward"] = {
        "mean": float(np.mean(total_returns)),
        "std": float(np.std(total_returns)),
    }

    # Compute the "env-equivalent" weighted reward from components
    # (should match total_reward closely in Phase 1, diverge in Phase 2)
    weighted = []
    for ep in per_episode:
        c = ep["components"]
        w = (
            ep["stand_weighted"]  # already 0.1 * Σ(stand_t · small_control_t)
            + 0.45 * c.get("door_openness_reward", 0)
            + 0.05 * c.get("door_hatch_openness_reward", 0)
            + 0.05 * c.get("hand_hatch_proximity_reward", 0)
            + 0.35 * c.get("passage_reward", 0)
        )
        weighted.append(w)
    agg["computed_weighted_reward"] = {
        "mean": float(np.mean(weighted)),
        "std": float(np.std(weighted)),
    }

    return {
        "per_episode": per_episode,
        "aggregate": agg,
        "success_rate": success_count / n_episodes,
        "n_episodes": n_episodes,
    }


# ────────────────────────────────────────────────────────────────────────────
# Pretty-print comparison
# ────────────────────────────────────────────────────────────────────────────

def print_comparison(results: dict):
    """Print a formatted comparison table for all evaluated models."""

    model_names = list(results.keys())
    n_models = len(model_names)

    # ── Header ──
    print("\n" + "=" * 90)
    print("REWARD DECOMPOSITION — PER-EPISODE ACCUMULATED COMPONENTS")
    print("(mean ± std across episodes)")
    print("=" * 90)

    # ── Success rate row ──
    print(f"\n{'Success Rate':<30}", end="")
    for name in model_names:
        sr = results[name]["success_rate"]
        print(f" | {name:<20}", end="")
    print()
    print(f"{'':<30}", end="")
    for name in model_names:
        sr = results[name]["success_rate"]
        print(f" | {sr:.3f}               ", end="")
    print()

    # ── Total reward row ──
    print(f"\n{'Total Reward (env)':<30}", end="")
    for name in model_names:
        tr = results[name]["aggregate"]["total_reward"]
        print(f" | {tr['mean']:>7.1f} ± {tr['std']:<7.1f}", end="")
    print()

    # ── Computed weighted reward (should ≈ env reward in Phase 1) ──
    print(f"{'Computed weighted reward':<30}", end="")
    for name in model_names:
        cw = results[name]["aggregate"]["computed_weighted_reward"]
        print(f" | {cw['mean']:>7.1f} ± {cw['std']:<7.1f}", end="")
    print()

    # ── Per-component rows ──
    print("\n" + "-" * 90)
    print("RAW COMPONENTS (unweighted per-episode sums):")
    print("-" * 90)

    for comp in REWARD_COMPONENTS:
        weight = COMPONENT_WEIGHTS[comp]
        label = f"  {comp}"
        print(f"\n{label:<30}", end="")
        print(f"  [×{weight}]", end="")
        # pad to align
        pad = 30 - len(label) - len(f"  [×{weight}]")
        print(" " * max(pad, 0), end="")
        for name in model_names:
            c = results[name]["aggregate"][comp]
            print(f" | {c['mean']:>7.3f} ± {c['std']:<7.3f}", end="")
        print()

    # ── Weighted contribution rows ──
    print("\n" + "-" * 90)
    print("WEIGHTED CONTRIBUTIONS (component × weight, per-episode):")
    print("-" * 90)

    for comp in WEIGHTED_TERMS:
        weight = COMPONENT_WEIGHTS[comp]
        label = f"  {comp}"
        print(f"\n{label:<30}", end="")
        print(f"  [×{weight}]", end="")
        pad = 30 - len(label) - len(f"  [×{weight}]")
        print(" " * max(pad, 0), end="")

        for name in model_names:
            raw_vals = [
                ep["components"].get(comp, 0.0)
                for ep in results[name]["per_episode"]
            ]
            if comp == "stand_reward":
                # stand_reward is multiplied by small_control PER STEP in the env,
                # so use the per-step accumulated product, not Σstand · Σsmall.
                weighted = [
                    ep["stand_weighted"]
                    for ep in results[name]["per_episode"]
                ]
            else:
                weighted = [weight * v for v in raw_vals]
            print(f" | {np.mean(weighted):>7.3f} ± {np.std(weighted):<7.3f}", end="")
        print()

    # ── Fraction of total weighted reward ──
    print("\n" + "-" * 90)
    print("FRACTION OF COMPUTED WEIGHTED REWARD (%):")
    print("-" * 90)

    for comp in WEIGHTED_TERMS:
        weight = COMPONENT_WEIGHTS[comp]
        print(f"  {comp:<35}", end="")

        for name in model_names:
            raw_vals = [
                ep["components"].get(comp, 0.0)
                for ep in results[name]["per_episode"]
            ]
            if comp == "stand_reward":
                weighted_comp = np.array([
                    ep["stand_weighted"]
                    for ep in results[name]["per_episode"]
                ])
            else:
                weighted_comp = np.array(raw_vals) * weight

            total_weighted = np.array([
                (
                    ep["stand_weighted"]
                    + 0.45 * ep["components"].get("door_openness_reward", 0)
                    + 0.05 * ep["components"].get("door_hatch_openness_reward", 0)
                    + 0.05 * ep["components"].get("hand_hatch_proximity_reward", 0)
                    + 0.35 * ep["components"].get("passage_reward", 0)
                )
                for ep in results[name]["per_episode"]
            ])

            # Per-episode fraction, then average (avoids div-by-total bias)
            fracs = np.divide(
                weighted_comp, total_weighted,
                out=np.zeros_like(weighted_comp),
                where=total_weighted > 0
            ) * 100.0
            print(f" | {np.mean(fracs):>6.1f}% ± {np.std(fracs):<6.1f}%", end="")
        print()

    print("\n" + "=" * 90)


# ────────────────────────────────────────────────────────────────────────────
# CSV export
# ────────────────────────────────────────────────────────────────────────────

def export_csv(results: dict, csv_path: str):
    """Export per-episode data to CSV for external analysis."""
    rows = []
    for model_name, result in results.items():
        for i, ep in enumerate(result["per_episode"]):
            row = {
                "model": model_name,
                "episode": i,
                "success": int(ep["success"]),
                "total_reward": ep["total_reward"],
                "stand_weighted": ep.get("stand_weighted", 0.0),
                "n_steps": ep["n_steps"],
            }
            for c in REWARD_COMPONENTS:
                row[c] = ep["components"].get(c, 0.0)
            rows.append(row)

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nPer-episode data exported to: {csv_path}")


# ────────────────────────────────────────────────────────────────────────────
# CLI
# ────────────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Reward decomposition analysis for Door task fine-tuning",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Phase 1 (dense reward) — most informative for reward decomposition
  python analyze_reward.py \\
      --baseline_exp curriculum-v7.1 \\
      --ppo_exp ft_ppo_v2 --ppo_model best_model \\
      --grpo_exp ft_grpo_v2 --grpo_model best_model \\
      --seeds 1024 2024 777 88 13 --n_episodes 50

  # Phase 2 (sparse oracle)
  python analyze_reward.py \\
      --baseline_exp curriculum-v7.1 \\
      --ppo_exp ft_ppo_v3 --grpo_exp ft_grpo_v3 \\
      --phase2 --seeds 1024 2024 777 88 13

  # All models at once + CSV export
  python analyze_reward.py \\
      --baseline_exp curriculum-v7.1 \\
      --ppo_exp ft_ppo_v2 --ppo_model best_by_reward \\
      --grpo_exp ft_grpo_v2 --grpo_model best_by_reward \\
      --ppo2_exp ft_ppo_v3 --ppo2_model best_by_reward \\
      --grpo2_exp ft_grpo_v3 --grpo2_model best_by_reward \\
      --phase2 --csv reward_decomp.csv
        """,
    )
    parser.add_argument("--env_name", type=str, default="h1hand-door-v0")
    parser.add_argument("--model_dir", type=str, default="models")

    # Baseline
    parser.add_argument("--baseline_exp", type=str, default="curriculum-v7.1",
                        help="Experiment name for the baseline model")
    parser.add_argument("--baseline_model", type=str, default="best_model",
                        help="Model file tag for baseline (without .zip)")
    parser.add_argument("--baseline_vecnorm", type=str, default="best_vecnormalize.pkl",
                        help="VecNormalize file for baseline")

    # PPO (Phase 1)
    parser.add_argument("--ppo_exp", type=str, default=None,
                        help="Experiment name for PPO fine-tuned model")
    parser.add_argument("--ppo_model", type=str, default="best_by_reward",
                        help="Model file tag for PPO")
    parser.add_argument("--ppo_algo", type=str, default="ppo")

    # GRPO (Phase 1)
    parser.add_argument("--grpo_exp", type=str, default=None,
                        help="Experiment name for GRPO fine-tuned model")
    parser.add_argument("--grpo_model", type=str, default="best_by_reward",
                        help="Model file tag for GRPO")
    parser.add_argument("--grpo_algo", type=str, default="grpo")

    # Optional second set of models (e.g. Phase 2)
    parser.add_argument("--ppo2_exp", type=str, default=None,
                        help="Experiment name for second PPO model (e.g. Phase 2)")
    parser.add_argument("--ppo2_model", type=str, default="best_by_reward")
    parser.add_argument("--grpo2_exp", type=str, default=None,
                        help="Experiment name for second GRPO model (e.g. Phase 2)")
    parser.add_argument("--grpo2_model", type=str, default="best_by_reward")

    # Evaluation config
    parser.add_argument("--seeds", type=int, nargs="+",
                        default=[1024, 2024, 777, 88, 13])
    parser.add_argument("--n_episodes", type=int, default=20)
    parser.add_argument("--phase2", action="store_true",
                        help="Flag for Phase 2 models (sparse oracle). "
                             "Reward decomp is less informative here "
                             "(env reward is binary), but the script will "
                             "still report what the info dict provides.")

    # Output
    parser.add_argument("--csv", type=str, default=None,
                        help="Export per-episode data to CSV file")

    return parser.parse_args()


# ────────────────────────────────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    # ── Build list of models to evaluate ──
    models_to_eval = []

    # Helper to add a model config
    def add_model(label: str, exp_name: str, model_tag: str,
                  algo: str, vecnorm_tag: str | None = None):
        if exp_name is None:
            return
        model_path = os.path.join(
            args.model_dir, args.env_name, exp_name, f"{model_tag}.zip"
        )
        if not os.path.exists(model_path):
            print(f"[SKIP] Model not found: {model_path}")
            return
        vn_tag = vecnorm_tag or f"{model_tag}_vecnormalize.pkl"
        vecnorm_path = os.path.join(
            args.model_dir, args.env_name, exp_name, vn_tag
        )
        if not os.path.exists(vecnorm_path):
            print(f"[WARN] VecNormalize not found: {vecnorm_path} — using fresh stats")
            vecnorm_path = None
        models_to_eval.append({
            "label": label,
            "model_path": model_path,
            "vecnorm_path": vecnorm_path,
            "algo": algo,
        })
        print(f"[LOAD] {label}: {model_path}")

    add_model("Baseline", args.baseline_exp, args.baseline_model,
              "ppo", args.baseline_vecnorm)
    add_model("PPO", args.ppo_exp, args.ppo_model, args.ppo_algo)
    add_model("GRPO", args.grpo_exp, args.grpo_model, args.grpo_algo)
    add_model("PPO-v3", args.ppo2_exp, args.ppo2_model, "ppo")
    add_model("GRPO-v3", args.grpo2_exp, args.grpo2_model, "grpo")

    if not models_to_eval:
        print("[ERROR] No models found to evaluate. Check experiment names.")
        sys.exit(1)

    print(f"\nPhase 2 mode: {args.phase2}")
    print(f"Evaluation seeds: {args.seeds}")
    print(f"Episodes per seed: {args.n_episodes}")
    print(f"Total episodes per model: {len(args.seeds) * args.n_episodes}")
    print("-" * 60)

    # ── Evaluate all models ──
    results = {}
    for cfg in models_to_eval:
        label = cfg["label"]
        print(f"\nEvaluating {label} ...")

        algo_cls = {"ppo": PPO, "grpo": GRPO}[cfg["algo"]]

        # Load model once, evaluate across seeds
        tmp_env = DummyVecEnv([make_env(args.env_name)])
        tmp_env = VecNormalize(tmp_env, norm_obs=True, norm_reward=False,
                              clip_obs=10.0, training=False)
        model = algo_cls.load(cfg["model_path"], env=tmp_env)
        tmp_env.close()

        # We run all episodes with seed rotation
        all_per_ep = []
        total_success = 0
        total_eps = 0

        for base_seed in args.seeds:
            r = eval_model(
                model=model,
                env_name=args.env_name,
                vecnormalize_path=cfg["vecnorm_path"],
                n_episodes=args.n_episodes,
                base_seed=base_seed,
            )
            all_per_ep.extend(r["per_episode"])
            total_success += r["success_rate"] * r["n_episodes"]
            total_eps += r["n_episodes"]

        # Recompute aggregate over all episodes
        agg = {}
        for c in REWARD_COMPONENTS:
            vals = [ep["components"].get(c, 0.0) for ep in all_per_ep]
            agg[c] = {
                "mean": float(np.mean(vals)),
                "std": float(np.std(vals)),
                "min": float(np.min(vals)),
                "max": float(np.max(vals)),
            }

        total_returns = [ep["total_reward"] for ep in all_per_ep]
        agg["total_reward"] = {
            "mean": float(np.mean(total_returns)),
            "std": float(np.std(total_returns)),
        }

        weighted = []
        for ep in all_per_ep:
            c = ep["components"]
            w = (
                ep["stand_weighted"]  # already 0.1 * Σ(stand_t · small_control_t)
                + 0.45 * c.get("door_openness_reward", 0)
                + 0.05 * c.get("door_hatch_openness_reward", 0)
                + 0.05 * c.get("hand_hatch_proximity_reward", 0)
                + 0.35 * c.get("passage_reward", 0)
            )
            weighted.append(w)
        agg["computed_weighted_reward"] = {
            "mean": float(np.mean(weighted)),
            "std": float(np.std(weighted)),
        }

        results[label] = {
            "per_episode": all_per_ep,
            "aggregate": agg,
            "success_rate": total_success / total_eps if total_eps > 0 else 0.0,
            "n_episodes": total_eps,
        }

    # ── Print comparison ──
    print_comparison(results)

    # ── Optional CSV export ──
    if args.csv:
        export_csv(results, args.csv)

    # ── Diagnosis ──
    print("\nDIAGNOSIS:")
    if "PPO" in results and "Baseline" in results:
        ppo_passage = results["PPO"]["aggregate"]["passage_reward"]["mean"]
        baseline_passage = results["Baseline"]["aggregate"]["passage_reward"]["mean"]
        ppo_door = results["PPO"]["aggregate"]["door_openness_reward"]["mean"]
        baseline_door = results["Baseline"]["aggregate"]["door_openness_reward"]["mean"]
        ppo_reward = results["PPO"]["aggregate"]["total_reward"]["mean"]
        baseline_reward = results["Baseline"]["aggregate"]["total_reward"]["mean"]

        print(f"  PPO total reward gain over baseline: {ppo_reward - baseline_reward:+.1f}")
        print(f"     passage_reward delta:         {ppo_passage - baseline_passage:+.3f}")
        print(f"     door_openness_reward delta:    {ppo_door - baseline_door:+.3f}")
        print(f"  → PPO's reward gain is primarily from: ", end="")
        if (ppo_passage - baseline_passage) > (ppo_door - baseline_door):
            print("passage_reward (robot moving forward through door)")
        else:
            print("door_openness_reward (door angle manipulation)")

    if "GRPO" in results and "Baseline" in results:
        grpo_reward = results["GRPO"]["aggregate"]["total_reward"]["mean"]
        print(f"  GRPO total reward change vs baseline: {grpo_reward - baseline_reward:+.1f}")
        if abs(grpo_reward - baseline_reward) < 30:
            print("  → GRPO reward is essentially unchanged from baseline")
        print("  → Consistent with the hypothesis that GRPO's group-relative")
        print("    advantage does not amplify reward-hacking behavior.")


if __name__ == "__main__":
    main()
