"""
KL divergence and policy-shift analysis for Door task fine-tuning.

Measures how far PPO and GRPO fine-tuned policies have moved from the
baseline pretrained policy, addressing the key question:
    "Is GRPO stable because it's robust, or because it barely moved?"

Metrics:
    1. Per-observation KL divergence: KL(π_finetuned || π_baseline)
       Closed-form for diagonal Gaussian policies.
    2. Deterministic action difference: ||μ_finetune - μ_baseline||
    3. Log-std shift: Δ log_std per action dimension
    4. Parameter-space L2 distance: per-layer breakdown
    5. Behavioral summary: mean action magnitude, action variance

Usage:
    conda activate humanoidbench

    # Phase 1 models
    python analyze_kl.py \
        --baseline_exp curriculum-v7.1 \
        --ppo_exp ft_ppo_v2 --ppo_model best_by_reward \
        --grpo_exp ft_grpo_v2 --grpo_model best_by_reward \
        --n_rollout_episodes 10 --n_obs 1000

    # Include Phase 2 models
    python analyze_kl.py \
        --baseline_exp curriculum-v7.1 \
        --ppo_exp ft_ppo_v2 --ppo_model best_by_reward \
        --grpo_exp ft_grpo_v2 --grpo_model best_by_reward \
        --ppo2_exp ft_ppo_v3 --ppo2_model best_by_reward \
        --grpo2_exp ft_grpo_v3 --grpo2_model best_by_reward
"""

import argparse
import os
import sys
from collections import OrderedDict

import numpy as np
import torch as th
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

import humanoid_bench
from utils.grpo import GRPO


# ────────────────────────────────────────────────────────────────────────────
# KL divergence helpers
# ────────────────────────────────────────────────────────────────────────────

def gaussian_kl(
    mean_p: np.ndarray,
    log_std_p: np.ndarray,
    mean_q: np.ndarray,
    log_std_q: np.ndarray,
) -> np.ndarray:
    """KL(P || Q) for diagonal Gaussian distributions, per observation.

    Args:
        mean_p:   (N, D) — target/student distribution
        log_std_p: (D,) or (N, D) — log std of P
        mean_q:   (N, D) — reference/teacher distribution
        log_std_q: (D,) or (N, D) — log std of Q

    Returns:
        kl: (N,) — KL divergence per observation in nats
    """
    var_p = np.exp(2.0 * log_std_p)  # (D,) or (N, D)
    var_q = np.exp(2.0 * log_std_q)

    # Per-dimension KL: log(σ_q/σ_p) + (σ_p² + (μ_p-μ_q)²)/(2σ_q²) - 1/2
    kl_per_dim = (
        log_std_q - log_std_p
        + (var_p + np.square(mean_p - mean_q)) / (2.0 * var_q)
        - 0.5
    )
    return kl_per_dim.sum(axis=-1)


def get_policy_stats(model, obs_normalized: np.ndarray) -> dict:
    """Extract action distribution parameters for a batch of observations.

    Args:
        model: SB3 PPO/GRPO model
        obs_normalized: (N, obs_dim) — already VecNormalize'd observations

    Returns:
        dict with keys:
            mean:     (N, act_dim) action means
            log_std:  (act_dim,)  log standard deviations
            actions:  (N, act_dim) sampled actions (deterministic = mean here)
    """
    obs_tensor = th.as_tensor(obs_normalized, dtype=th.float32).to(model.device)

    with th.no_grad():
        # SB3 PPO policy forward pass
        # model.policy(obs) returns (actions, values, log_probs)
        # But we need the distribution parameters directly
        features = model.policy.extract_features(obs_tensor)
        if hasattr(model.policy, 'mlp_extractor'):
            latent = model.policy.mlp_extractor.forward_actor(features)
        else:
            latent = features
        mean = model.policy.action_net(latent)

        if hasattr(model.policy, 'log_std'):
            log_std = model.policy.log_std.detach().cpu().numpy()
        else:
            # Fallback: some architectures store it differently
            log_std = model.policy.action_dist.log_std.detach().cpu().numpy()

    return {
        "mean": mean.detach().cpu().numpy(),
        "log_std": log_std,
    }


# ────────────────────────────────────────────────────────────────────────────
# Observation collection
# ────────────────────────────────────────────────────────────────────────────

class ObsCollectorWrapper(gym.Wrapper):
    """Minimal wrapper that just passes through, collecting raw observations."""

    def __init__(self, env):
        super().__init__(env)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return obs, info


def make_collector_env(env_name: str = "h1hand-door-v0"):
    def _init():
        env = gym.make(env_name)
        env = ObsCollectorWrapper(env)
        return env
    return _init


def collect_observations(
    model,
    env_name: str,
    vecnormalize_path: str,
    n_episodes: int = 10,
    base_seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Run the baseline model and collect raw observations + normalization stats.

    We build the env WITHOUT VecNormalize, manually apply the baseline's
    normalization before each model.predict(), and store the RAW observations.
    This lets each model later apply its own VecNormalize without
    double-normalization issues.

    Returns:
        raw_obs:   (N, obs_dim) un-normalized observations
        obs_mean:  (obs_dim,)   baseline VecNormalize mean
        obs_var:   (obs_dim,)   baseline VecNormalize variance
        eps:       float        baseline VecNormalize epsilon
    """
    # Build env WITHOUT VecNormalize to get raw observations
    env = DummyVecEnv([make_collector_env(env_name)])

    # Load baseline VecNormalize stats manually
    if vecnormalize_path and os.path.exists(vecnormalize_path):
        vn = VecNormalize.load(vecnormalize_path, env)
        obs_mean = vn.obs_rms.mean.copy()
        obs_var = vn.obs_rms.var.copy()
        eps = vn.epsilon
        vn.training = False
        vn.norm_reward = False
    else:
        obs_mean = np.zeros(env.observation_space.shape[0])
        obs_var = np.ones(env.observation_space.shape[0])
        eps = 1e-8

    model.set_env(env)

    all_raw_obs = []

    for ep in range(n_episodes):
        env.envs[0].reset(seed=base_seed + ep)
        raw_obs = env.reset()  # (1, obs_dim) — raw, not normalized
        done = False

        while not done:
            # Store the RAW observation
            all_raw_obs.append(raw_obs[0].copy())

            # Normalize manually before feeding to model
            obs_norm = (raw_obs - obs_mean) / np.sqrt(obs_var + eps)
            action, _ = model.predict(obs_norm, deterministic=True)
            raw_obs, _, dones, _ = env.step(action)
            done = bool(dones[0])

    env.close()
    return np.stack(all_raw_obs, axis=0), obs_mean, obs_var, eps


# ────────────────────────────────────────────────────────────────────────────
# Parameter distance helpers
# ────────────────────────────────────────────────────────────────────────────

def get_policy_params(model) -> dict:
    """Extract policy-relevant parameters (actor only, no critic)."""
    state = model.policy.state_dict()
    policy_params = OrderedDict()
    for name, param in state.items():
        # Filter to actor/policy parameters only
        if any(k in name for k in [
            "mlp_extractor.policy_net",
            "action_net",
            "log_std",
        ]):
            policy_params[name] = param.detach().cpu().numpy()
    return policy_params


def param_l2_distance(params_a: dict, params_b: dict) -> dict:
    """Compute per-layer and total L2 distance between two param dicts."""
    per_layer = OrderedDict()
    total_sq = 0.0
    total_n = 0

    for name in params_a:
        if name in params_b:
            diff = params_a[name].ravel() - params_b[name].ravel()
            sq = float(np.sum(np.square(diff)))
            per_layer[name] = float(np.sqrt(sq))
            total_sq += sq
            total_n += diff.size
        else:
            per_layer[name] = float("nan")

    return {
        "per_layer": per_layer,
        "total_l2": float(np.sqrt(total_sq)),
    }


# ────────────────────────────────────────────────────────────────────────────
# CLI
# ────────────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="KL divergence and policy-shift analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
python analyze_kl.py \
    --baseline_exp curriculum-v7.1 \
    --ppo_exp ft_ppo_v2_5075 --ppo_model best_by_reward \
    --grpo_exp ft_grpokl_5075 --grpo_model best_by_reward \
    --n_rollout_episodes 10
        """,
    )
    parser.add_argument("--env_name", type=str, default="h1hand-door-v0")
    parser.add_argument("--model_dir", type=str, default="models")

    # Baseline
    parser.add_argument("--baseline_exp", type=str, default="curriculum-v7.1")
    parser.add_argument("--baseline_model", type=str, default="best_model")
    parser.add_argument("--baseline_vecnorm", type=str, default="best_vecnormalize.pkl")

    # PPO Phase 1
    parser.add_argument("--ppo_exp", type=str, default=None)
    parser.add_argument("--ppo_model", type=str, default="best_model")
    parser.add_argument("--ppo_algo", type=str, default="ppo")

    # GRPO Phase 1
    parser.add_argument("--grpo_exp", type=str, default=None)
    parser.add_argument("--grpo_model", type=str, default="best_model")
    parser.add_argument("--grpo_algo", type=str, default="grpo")

    # Optional Phase 2 models
    parser.add_argument("--ppo2_exp", type=str, default=None)
    parser.add_argument("--ppo2_model", type=str, default="best_by_reward")
    parser.add_argument("--grpo2_exp", type=str, default=None)
    parser.add_argument("--grpo2_model", type=str, default="best_by_reward")

    # Collection
    parser.add_argument("--n_rollout_episodes", type=int, default=10)
    parser.add_argument("--n_obs", type=int, default=1000,
                        help="Max observations to use (subsample if more collected)")
    parser.add_argument("--collection_seed", type=int, default=0)

    return parser.parse_args()


# ────────────────────────────────────────────────────────────────────────────
# Model loading
# ────────────────────────────────────────────────────────────────────────────

def load_model(algo: str, model_path: str, env_name: str):
    """Load an SB3 model with a fresh dummy env."""
    algo_cls = {"ppo": PPO, "grpo": GRPO}[algo]
    tmp_env = DummyVecEnv([make_collector_env(env_name)])
    tmp_env = VecNormalize(tmp_env, norm_obs=True, norm_reward=False,
                           clip_obs=10.0, training=False)
    model = algo_cls.load(model_path, env=tmp_env)
    tmp_env.close()
    return model


def resolve_model_path(model_dir, env_name, exp_name, model_tag):
    model_path = os.path.join(model_dir, env_name, exp_name, f"{model_tag}.zip")
    if not os.path.exists(model_path):
        return None
    return model_path


def resolve_vecnorm_path(model_dir, env_name, exp_name, model_tag):
    """Resolve VecNormalize path. Tries {model_tag}_vecnormalize.pkl first,
    then best_vecnormalize.pkl as fallback (for v2 models)."""
    vn_path = os.path.join(model_dir, env_name, exp_name,
                           f"{model_tag}_vecnormalize.pkl")
    if os.path.exists(vn_path):
        return vn_path
    # Fallback: some older experiments use this naming
    fallback = os.path.join(model_dir, env_name, exp_name, "best_vecnormalize.pkl")
    if os.path.exists(fallback):
        return fallback
    return None


# ────────────────────────────────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    # ── Resolve models ──
    model_specs = []

    def add_spec(label, exp_name, model_tag, algo):
        if exp_name is None:
            return
        mp = resolve_model_path(args.model_dir, args.env_name, exp_name, model_tag)
        if mp is None:
            print(f"[SKIP] {label}: model not found ({exp_name}/{model_tag})")
            return
        vp = resolve_vecnorm_path(args.model_dir, args.env_name, exp_name, model_tag)
        model_specs.append({
            "label": label,
            "model_path": mp,
            "vecnorm_path": vp,
            "algo": algo,
        })

    add_spec("Baseline", args.baseline_exp, args.baseline_model, "ppo")
    add_spec("PPO", args.ppo_exp, args.ppo_model, args.ppo_algo)
    add_spec("GRPO", args.grpo_exp, args.grpo_model, args.grpo_algo)
    add_spec("PPO-v3", args.ppo2_exp, args.ppo2_model, "ppo")
    add_spec("GRPO-v3", args.grpo2_exp, args.grpo2_model, "grpo")

    if len(model_specs) < 2:
        print("[ERROR] Need at least 2 models to compare.")
        sys.exit(1)

    print("Models to compare:")
    for s in model_specs:
        print(f"  {s['label']:12s}  {s['model_path']}")
    print(f"\nCollection: {args.n_rollout_episodes} episodes, "
          f"max {args.n_obs} observations")
    print("-" * 60)

    # ── Step 1: Collect RAW observations using the baseline model ──
    print("\n[1/4] Collecting raw observations using baseline policy ...")
    baseline_spec = model_specs[0]
    assert baseline_spec["label"] == "Baseline", "First model must be Baseline"

    collector = load_model(
        baseline_spec["algo"],
        baseline_spec["model_path"],
        args.env_name,
    )
    raw_obs, baseline_obs_mean, baseline_obs_var, baseline_obs_eps = collect_observations(
        model=collector,
        env_name=args.env_name,
        vecnormalize_path=baseline_spec["vecnorm_path"],
        n_episodes=args.n_rollout_episodes,
        base_seed=args.collection_seed,
    )
    del collector

    # Subsample if needed
    if len(raw_obs) > args.n_obs:
        rng = np.random.RandomState(42)
        idx = rng.choice(len(raw_obs), args.n_obs, replace=False)
        raw_obs = raw_obs[idx]

    print(f"  Collected {len(raw_obs)} raw observations "
          f"(obs_dim={raw_obs.shape[1]})")

    # Also save baseline-normalized observations for reference
    baseline_obs_norm = (raw_obs - baseline_obs_mean) / np.sqrt(baseline_obs_var + baseline_obs_eps)

    # ── Step 2: Load all models, normalize with each model's own stats ──
    print("\n[2/4] Loading models and extracting policy distributions ...")

    policy_stats = {}

    for spec in model_specs:
        label = spec["label"]
        print(f"    {label} ...", end=" ", flush=True)

        model = load_model(spec["algo"], spec["model_path"], args.env_name)

        # Apply THIS model's VecNormalize to the raw observations
        if spec["vecnorm_path"] is not None:
            vn = VecNormalize.load(
                spec["vecnorm_path"],
                DummyVecEnv([make_collector_env(args.env_name)]),
            )
            vn.training = False
            vn.norm_reward = False
            obs_norm = (raw_obs - vn.obs_rms.mean) / np.sqrt(
                vn.obs_rms.var + vn.epsilon
            )
        else:
            obs_norm = baseline_obs_norm  # fallback to baseline normalization

        stats = get_policy_stats(model, obs_norm)
        policy_stats[label] = {
            "mean": stats["mean"],
            "log_std": stats["log_std"],
        }

        spec["params"] = get_policy_params(model)
        spec["log_std"] = stats["log_std"]

        del model
        print(f"done (mean shape: {stats['mean'].shape})")

    baseline_act_mean = policy_stats["Baseline"]["mean"]
    baseline_act_log_std = policy_stats["Baseline"]["log_std"]

    # ── Step 3: Compute KL divergences ──
    print("\n[3/4] Computing KL divergences and action differences ...\n")

    kl_results = {}
    action_diff_results = {}
    log_std_results = {}

    for spec in model_specs:
        label = spec["label"]
        if label == "Baseline":
            continue

        mean_p = policy_stats[label]["mean"]
        log_std_p = policy_stats[label]["log_std"]

        # KL(Finetuned || Baseline)
        kl_per_obs = gaussian_kl(mean_p, log_std_p, baseline_act_mean, baseline_act_log_std)

        kl_results[label] = {
            "mean": float(np.mean(kl_per_obs)),
            "median": float(np.median(kl_per_obs)),
            "std": float(np.std(kl_per_obs)),
            "p90": float(np.percentile(kl_per_obs, 90)),
            "p95": float(np.percentile(kl_per_obs, 95)),
            "max": float(np.max(kl_per_obs)),
            "min": float(np.min(kl_per_obs)),
        }

        # ||μ_finetune - μ_baseline|| per observation
        action_diff = np.linalg.norm(mean_p - baseline_act_mean, axis=-1)
        action_diff_results[label] = {
            "mean": float(np.mean(action_diff)),
            "median": float(np.median(action_diff)),
            "std": float(np.std(action_diff)),
            "p90": float(np.percentile(action_diff, 90)),
            "p95": float(np.percentile(action_diff, 95)),
            "max": float(np.max(action_diff)),
        }

        # Log-std shift per dimension
        delta_log_std = log_std_p - baseline_act_log_std
        log_std_results[label] = {
            "per_dim": delta_log_std.tolist(),
            "mean_delta": float(np.mean(delta_log_std)),
            "max_delta": float(np.max(np.abs(delta_log_std))),
        }

    # ── Print KL results ──
    print("=" * 80)
    print("KL DIVERGENCE ANALYSIS")
    print("=" * 80)

    header = f"{'Model':<12} {'Mean KL':>10} {'Median':>10} {'Std':>10} {'P90':>10} {'P95':>10} {'Max':>10}"
    print(f"\nKL(π_finetuned || π_baseline) in nats:")
    print("-" * len(header))
    print(header)
    print("-" * len(header))
    for label, kr in kl_results.items():
        print(f"{label:<12} {kr['mean']:>10.4f} {kr['median']:>10.4f} "
              f"{kr['std']:>10.4f} {kr['p90']:>10.4f} {kr['p95']:>10.4f} "
              f"{kr['max']:>10.4f}")

    # ── Print action difference results ──
    header2 = f"{'Model':<12} {'Mean':>10} {'Median':>10} {'P90':>10} {'P95':>10} {'Max':>10}"
    print(f"\n||μ_finetune - μ_baseline|| (L2 action difference):")
    print("-" * len(header2))
    print(header2)
    print("-" * len(header2))
    for label, ad in action_diff_results.items():
        print(f"{label:<12} {ad['mean']:>10.4f} {ad['median']:>10.4f} "
              f"{ad['p90']:>10.4f} {ad['p95']:>10.4f} {ad['max']:>10.4f}")

    # ── Print log-std shift ──
    print(f"\nΔ log_std from baseline (positive = higher entropy):")
    print(f"{'Model':<12} {'Mean Δ':>10} {'Max |Δ|':>10}  Per-dimension")
    print("-" * 60)
    for label, ls in log_std_results.items():
        dims_str = " ".join(f"{d:+.4f}" for d in ls["per_dim"][:10])
        if len(ls["per_dim"]) > 10:
            dims_str += f" ... ({len(ls['per_dim'])} dims total)"
        print(f"{label:<12} {ls['mean_delta']:>+10.4f} {ls['max_delta']:>10.4f}  {dims_str}")

    # ── Step 4: Parameter distance ──
    print("\n[4/4] Computing parameter-space distances ...\n")
    print("=" * 80)
    print("PARAMETER-SPACE DISTANCE")
    print("=" * 80)

    baseline_params = model_specs[0]["params"]

    # Find common keys across all models
    all_layers = set()
    for spec in model_specs:
        if "params" in spec and spec["label"] != "Baseline":
            all_layers.update(spec["params"].keys())

    layer_order = [
        "mlp_extractor.policy_net.0.weight",
        "mlp_extractor.policy_net.0.bias",
        "mlp_extractor.policy_net.2.weight",
        "mlp_extractor.policy_net.2.bias",
        "action_net.weight",
        "action_net.bias",
        "log_std",
    ]
    ordered_layers = [l for l in layer_order if l in all_layers]
    remaining = sorted(all_layers - set(ordered_layers))
    ordered_layers.extend(remaining)

    # Header
    non_baseline_labels = [s["label"] for s in model_specs if s["label"] != "Baseline"]
    col_w = max(14, max(len(l) for l in non_baseline_labels))
    print(f"\n{'Layer':<40}", end="")
    for label in non_baseline_labels:
        print(f" | {label:>{col_w}}", end="")
    print()
    print("-" * (40 + (col_w + 3) * len(non_baseline_labels)))

    total_dists = {}
    for label in non_baseline_labels:
        spec = [s for s in model_specs if s["label"] == label][0]
        dist_info = param_l2_distance(spec["params"], baseline_params)
        total_dists[label] = dist_info["total_l2"]
        spec["_param_dist"] = dist_info

    for layer_name in ordered_layers:
        print(f"{layer_name:<40}", end="")
        for label in non_baseline_labels:
            spec = [s for s in model_specs if s["label"] == label][0]
            val = spec["_param_dist"]["per_layer"].get(layer_name, float("nan"))
            if np.isnan(val):
                print(f" | {'N/A':>{col_w}}", end="")
            else:
                print(f" | {val:>{col_w}.4f}", end="")
        print()

    # Total row
    print("-" * (40 + (col_w + 3) * len(non_baseline_labels)))
    print(f"{'TOTAL (all policy params)':<40}", end="")
    for label in non_baseline_labels:
        print(f" | {total_dists[label]:>{col_w}.4f}", end="")
    print()

    # Ratio row (PPO / GRPO)
    if "PPO" in total_dists and "GRPO" in total_dists:
        ratio = total_dists["PPO"] / max(total_dists["GRPO"], 1e-10)
        print(f"{'Ratio PPO / GRPO':<40}", end="")
        for label in non_baseline_labels:
            if label in ("PPO", "GRPO"):
                print(f" | {'':>{col_w}}", end="")
            else:
                print(f" | {'':>{col_w}}", end="")
        print(f"  ← {ratio:.2f}×")

    # ── Diagnosis ──
    print("\n" + "=" * 80)
    print("DIAGNOSIS")
    print("=" * 80)

    # Report VecNormalize drift
    print("\n  VecNormalize observation statistics drift:")
    for spec in model_specs:
        label = spec["label"]
        if label == "Baseline":
            continue
        if spec["vecnorm_path"] is not None:
            vn = VecNormalize.load(
                spec["vecnorm_path"],
                DummyVecEnv([make_collector_env(args.env_name)]),
            )
            vn.training = False
            mean_diff = np.linalg.norm(vn.obs_rms.mean - baseline_obs_mean)
            var_diff = np.linalg.norm(vn.obs_rms.var - baseline_obs_var)
            print(f"    {label:<12s} ||Δ mean||={mean_diff:.4f}  ||Δ var||={var_diff:.4f}")

    # Determine the key comparison pair (Phase 1 PPO vs GRPO)
    if "PPO" in kl_results and "GRPO" in kl_results:
        ppo_kl = kl_results["PPO"]["mean"]
        grpo_kl = kl_results["GRPO"]["mean"]
        ppo_param = total_dists.get("PPO", 0)
        grpo_param = total_dists.get("GRPO", 0)

        print(f"""
  KL(PPO || Baseline)        = {ppo_kl:.4f} nats
  KL(GRPO || Baseline)       = {grpo_kl:.4f} nats
  KL ratio (PPO / GRPO)      = {ppo_kl / max(grpo_kl, 1e-10):.2f}×

  Param distance PPO         = {ppo_param:.4f}
  Param distance GRPO        = {grpo_param:.4f}
  Param ratio (PPO / GRPO)   = {ppo_param / max(grpo_param, 1e-10):.2f}×
""")

        # Interpret
        kl_threshold = 0.1   # nats — below this, the policy is nearly unchanged
        param_ratio_threshold = 2.0  # PPO moved at least 2× more than GRPO

        if grpo_kl < kl_threshold:
            print("  ⚠  GRPO KL is very small (< 0.1 nats).")
            print("     GRPO's action distribution is nearly identical to baseline.")
            print("     → GRPO's 'stability' may be because it barely updated,")
            print("       not because it learned a robust alternative strategy.")
        elif grpo_kl < ppo_kl * 0.5:
            print("  ✓  GRPO has shifted from baseline, but PPO shifted much more.")
            print("     This is consistent with GRPO being conservative but not stuck.")
        else:
            print("  ✓  Both models have meaningfully shifted from baseline.")
            print("     The difference is in the DIRECTION of the shift, not the magnitude.")

        if ppo_param > grpo_param * param_ratio_threshold:
            print(f"  ⚠  PPO parameter change is {ppo_param/grpo_param:.1f}× larger than GRPO.")
            print("     PPO is making much larger weight updates — possibly overfitting.")
        elif grpo_param < ppo_param * 0.5:
            print("  ✓  GRPO parameter change is notably smaller than PPO.")
        else:
            print("  → Both made comparable parameter changes.")

    else:
        print("  (Add --ppo_exp and --grpo_exp for automated diagnosis.)")

    print("\nDone.")


if __name__ == "__main__":
    main()
