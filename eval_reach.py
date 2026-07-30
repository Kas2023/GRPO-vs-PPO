"""
Multi-seed evaluation on the HumanoidBench Reach task (h1hand-reach-v0).

Loads a fine-tuned SB3 model (PPO/GRPO) or the raw pretrained reach model
(torch_model.pt) and evaluates across multiple seeds.  Reports the same
metrics as eval.py plus reach-specific ones.

Metrics:
    - Success rate       : hand_dist < 0.05 m at any point in the episode
    - Mean return        : cumulative reward
    - Episode length
    - Action smoothness  : cumulative ||a_t - a_{t-1}||^2 on body actions
    - Min hand distance  : closest the left hand got to the target

Usage:
    # Evaluate a fine-tuned model
    python eval_reach.py \\
        --exp_name reach_grpo \\
        --model_file best_by_success.zip \\
        --vecnormalize_file best_by_success_vecnormalize.pkl \\
        --algo grpo \\
        --seeds 42 2024 777 88 13 \\
        --n_eval_episodes 20

    # Evaluate the raw pretrained reach model (baseline)
    python eval_reach.py \\
        --exp_name baseline \\
        --baseline \\
        --seeds 42 2024 777 88 13 \\
        --n_eval_episodes 20
"""

import argparse
import os
import sys
import numpy as np
import torch as th
import gymnasium as gym
from gymnasium.spaces import Box

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

import mujoco
import humanoid_bench
from humanoid_bench.wrappers import get_body_idxs

from utils.grpo import GRPO

# ──────────────────────────────────────────────────────────────────────
# Observation wrapper (same as test_reach.py)
# ──────────────────────────────────────────────────────────────────────

class ReachObsWrapper(gym.ObservationWrapper):
    """Transform full 155D reach-task observation into the 55D format
    expected by the pretrained reach model."""

    def __init__(self, env):
        super().__init__(env)
        model = env.unwrapped.model
        self.body_idxs, self.body_vel_idxs = get_body_idxs(model)
        self.robot_dof = env.unwrapped.robot.dof
        self.observation_space = Box(
            low=-np.inf, high=np.inf, shape=(55,), dtype=np.float64
        )

    def observation(self, obs: np.ndarray) -> np.ndarray:
        robot_dof = self.robot_dof
        position = obs[:robot_dof]
        velocity = obs[robot_dof:robot_dof * 2 - 1]
        left_hand = obs[robot_dof * 2 - 1:robot_dof * 2 + 2]
        target = obs[robot_dof * 2 + 2:]

        body_pos = position[self.body_idxs]
        body_vel = velocity[self.body_vel_idxs]

        offset = np.array([body_pos[0], body_pos[1], 0.0])
        body_pos[:3] = body_pos[:3] - offset
        left_hand = left_hand - offset
        target = target - offset

        return np.concatenate([body_pos[2:], body_vel, left_hand, target])


# ──────────────────────────────────────────────────────────────────────
# Action wrapper (same as test_reach.py)
# ──────────────────────────────────────────────────────────────────────

class ReachActWrapper(gym.ActionWrapper):
    """Expand 19D body-joint actions to 61D (body + fixed hands)."""

    def __init__(self, env):
        super().__init__(env)
        self.action_space = Box(low=-1, high=1, shape=(19,), dtype=np.float32)
        if env.unwrapped.model.nu > 19:
            self.act_idxs = list(range(15)) + list(range(16, 20))
        else:
            self.act_idxs = list(range(19))
        self._full_nu = env.unwrapped.model.nu

    def action(self, action: np.ndarray) -> np.ndarray:
        if self._full_nu <= 19:
            return action
        env = self.env.unwrapped
        body_action = (action + 1) / 2 * (
            env.action_high[self.act_idxs] - env.action_low[self.act_idxs]
        ) + env.action_low[self.act_idxs]
        full_action = env.data.ctrl.copy()
        full_action[self.act_idxs] = body_action
        full_action[15] = 1.57
        full_action[20] = 1.57
        return 2 * (full_action - env.action_low) / (env.action_high - env.action_low) - 1


# ──────────────────────────────────────────────────────────────────────
# Metrics wrapper (same logic as test_reach.py)
# ──────────────────────────────────────────────────────────────────────

class ReachMetricsWrapper(gym.Wrapper):
    """Tracks hand distance, success, and action smoothness."""

    def __init__(self, env):
        super().__init__(env)
        self._episode_success = False
        self._prev_action = None
        self._episode_action_sqdiff = 0.0
        self._min_hand_dist = float("inf")

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._episode_success = False
        self._prev_action = None
        self._episode_action_sqdiff = 0.0
        self._min_hand_dist = float("inf")
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        hand_dist = info.get("hand_dist", None)
        if hand_dist is not None and hand_dist < self._min_hand_dist:
            self._min_hand_dist = hand_dist
        if hand_dist is not None and hand_dist < 0.05:
            self._episode_success = True

        if self._prev_action is not None:
            self._episode_action_sqdiff += float(
                np.sum(np.square(action - self._prev_action))
            )
        self._prev_action = action.copy()

        if terminated or truncated:
            if self._episode_success:
                info["success"] = 1
            info["action_smoothness"] = self._episode_action_sqdiff
            info["min_hand_dist"] = self._min_hand_dist

        return obs, reward, terminated, truncated, info


# ──────────────────────────────────────────────────────────────────────
# Environment factory
# ──────────────────────────────────────────────────────────────────────

def make_env(seed: int = 0):
    """Create a wrapped reach env for evaluation (no Monitor needed)."""
    def _init():
        env = gym.make("h1hand-reach-v0", render_mode="rgb_array")
        env = ReachObsWrapper(env)
        env = ReachActWrapper(env)
        env = ReachMetricsWrapper(env)
        env.reset(seed=seed)
        return env
    return _init


# ──────────────────────────────────────────────────────────────────────
# Weight loading (same as test_reach.py)
# ──────────────────────────────────────────────────────────────────────

def load_reach_weights_to_policy(model, pt_path: str, mean_path: str, var_path: str):
    """Load pretrained reach TorchModel weights into an SB3 policy."""
    pt_weights = th.load(pt_path, map_location="cpu")
    policy_state = model.policy.state_dict()

    mapping = {
        "dense1.weight": "mlp_extractor.policy_net.0.weight",
        "dense1.bias":   "mlp_extractor.policy_net.0.bias",
        "dense2.weight": "mlp_extractor.policy_net.2.weight",
        "dense2.bias":   "mlp_extractor.policy_net.2.bias",
        "dense3.weight": "action_net.weight",
        "dense3.bias":   "action_net.bias",
    }

    for pt_key, sb3_key in mapping.items():
        if pt_key in pt_weights and sb3_key in policy_state:
            if pt_weights[pt_key].shape == policy_state[sb3_key].shape:
                policy_state[sb3_key] = pt_weights[pt_key]

    model.policy.load_state_dict(policy_state)

    # VecNormalize stats
    pt_mean = np.load(mean_path)[0].astype(np.float64)
    pt_var = np.load(var_path)[0].astype(np.float64)
    train_env = model.get_vec_normalize_env()
    if train_env is not None:
        train_env.obs_rms.mean = pt_mean.copy()
        train_env.obs_rms.var = pt_var.copy()
        train_env.obs_rms.count = 1e4


# ──────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_name", type=str, required=True,
                        help="Experiment name (subdirectory under models/h1hand-reach-v0/)")
    parser.add_argument("--model_dir", type=str, default="models/h1hand-reach-v0")
    parser.add_argument("--model_file", type=str, default="best_by_success.zip")
    parser.add_argument("--vecnormalize_file", type=str,
                        default="best_by_success_vecnormalize.pkl")
    parser.add_argument("--algo", type=str, default="ppo", choices=["ppo", "grpo"])
    parser.add_argument("--seeds", type=int, nargs="+",
                        default=[42, 1024, 2024, 777, 88, 13])
    parser.add_argument("--n_eval_episodes", type=int, default=20)
    parser.add_argument("--baseline", action="store_true",
                        help="Evaluate the raw pretrained reach model "
                             "(torch_model.pt) as baseline, ignoring --exp_name "
                             "and --model_file.")
    parser.add_argument("--reach_model_dir", type=str,
                        default="humanoid-bench/data/reach_one_hand",
                        help="Path to reach_one_hand data (for --baseline)")
    return parser.parse_args()


# ──────────────────────────────────────────────────────────────────────
# Single-seed evaluation
# ──────────────────────────────────────────────────────────────────────

def eval_single_seed(
    model,
    vecnormalize_path: str | None,
    obs_mean: np.ndarray | None = None,
    obs_var: np.ndarray | None = None,
    n_eval_episodes: int = 20,
    base_seed: int = 0,
):
    """Run evaluation with a specific base seed.

    Parameters
    ----------
    obs_mean, obs_var : optional explicit VecNormalize stats (used in
        baseline mode where the model's attached env differs from the
        eval env being built here).
    """

    env = DummyVecEnv([make_env(seed=base_seed)])
    env = VecNormalize(env, norm_obs=True, norm_reward=False,
                       clip_obs=10.0, training=False)

    # Load VecNormalize stats
    if vecnormalize_path and os.path.exists(vecnormalize_path):
        env = VecNormalize.load(vecnormalize_path, env)
        env.training = False
        env.norm_reward = False
    elif obs_mean is not None and obs_var is not None:
        env.obs_rms.mean = obs_mean.copy()
        env.obs_rms.var = obs_var.copy()
        env.obs_rms.count = 1e4

    model.set_env(env)

    success_count = 0
    episode_returns = []
    episode_lengths = []
    episode_action_smoothness = []
    episode_min_hand_dists = []

    for ep in range(n_eval_episodes):
        # Re-seed the underlying gym env for independent rollouts
        env.venv.envs[0].reset(seed=base_seed + ep)
        obs = env.reset()
        done = False
        ep_return = 0.0
        ep_length = 0
        ep_success = False
        ep_min_dist = float("inf")

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, dones, info = env.step(action)
            ep_return += float(reward[0])
            ep_length += 1

            env_info = info[0]
            if env_info.get("success", 0) == 1:
                ep_success = True
            d = env_info.get("min_hand_dist", float("inf"))
            if d < ep_min_dist:
                ep_min_dist = d

            done = bool(dones[0])

        if ep_success:
            success_count += 1
        episode_returns.append(ep_return)
        episode_lengths.append(ep_length)
        episode_action_smoothness.append(
            info[0].get("action_smoothness", 0.0)
        )
        episode_min_hand_dists.append(ep_min_dist)

    env.close()

    return {
        "seed": base_seed,
        "success_rate": success_count / n_eval_episodes,
        "success_count": success_count,
        "mean_return": float(np.mean(episode_returns)),
        "std_return": float(np.std(episode_returns)),
        "mean_length": float(np.mean(episode_lengths)),
        "std_length": float(np.std(episode_lengths)),
        "mean_action_smoothness": float(np.mean(episode_action_smoothness)),
        "mean_min_hand_dist": float(np.mean(episode_min_hand_dists)),
    }


# ──────────────────────────────────────────────────────────────────────
# Multi-seed evaluation
# ──────────────────────────────────────────────────────────────────────

def eval_multi_seed(args):
    env_name = "h1hand-reach-v0"

    if args.baseline:
        # ── Baseline mode: load raw torch_model.pt into fresh SB3 ──
        pt_path = os.path.join(args.reach_model_dir, "torch_model.pt")
        mean_path = os.path.join(args.reach_model_dir, "mean.npy")
        var_path = os.path.join(args.reach_model_dir, "var.npy")

        for p in [pt_path, mean_path, var_path]:
            if not os.path.exists(p):
                print(f"[ERROR] Missing: {p}")
                sys.exit(1)

        print("=" * 60)
        print("BASELINE — Raw pretrained reach model (torch_model.pt)")
        print("=" * 60)
        print(f"Reach model : {pt_path}")
        print(f"Task        : {env_name}")
        print(f"Seeds       : {args.seeds}")
        print(f"Episodes    : {args.n_eval_episodes} per seed")

        # Create a dummy VecEnv just to initialise the model
        dummy_env = DummyVecEnv([make_env(seed=0)])
        dummy_env = VecNormalize(dummy_env, norm_obs=True, norm_reward=True,
                                 clip_obs=10.0)

        algo_cls = {"ppo": PPO, "grpo": GRPO}[args.algo]
        model = algo_cls(
            "MlpPolicy",
            dummy_env,
            policy_kwargs=dict(net_arch=dict(pi=[256, 256], vf=[256, 256])),
            device="cuda",
        )
        load_reach_weights_to_policy(model, pt_path, mean_path, var_path)

        # Capture VecNormalize stats before the dummy env is discarded
        train_norm = model.get_vec_normalize_env()
        obs_mean = train_norm.obs_rms.mean.copy()
        obs_var = train_norm.obs_rms.var.copy()
        dummy_env.close()

        vecnorm_path = None
        model_path = "(raw torch_model.pt)"

    else:
        # ── Fine-tuned model mode ──
        obs_mean = obs_var = None  # not used; stats come from vecnorm file
        model_path = os.path.join(args.model_dir, args.exp_name, args.model_file)
        vecnorm_path = os.path.join(args.model_dir, args.exp_name,
                                    args.vecnormalize_file)

        if not os.path.exists(model_path):
            print(f"[ERROR] Model not found: {model_path}")
            sys.exit(1)

        print("=" * 60)
        print(f"Fine-tuned model: {args.exp_name}")
        print("=" * 60)
        print(f"Model       : {model_path}")
        print(f"VecNorm     : {vecnorm_path}")
        print(f"Algorithm   : {args.algo}")
        print(f"Task        : {env_name}")
        print(f"Seeds       : {args.seeds}")
        print(f"Episodes    : {args.n_eval_episodes} per seed")

        algo_cls = {"ppo": PPO, "grpo": GRPO}[args.algo]
        # Temporary env just for loading
        tmp_env = DummyVecEnv([make_env(seed=0)])
        tmp_env = VecNormalize(tmp_env, norm_obs=True, norm_reward=False,
                               clip_obs=10.0, training=False)
        model = algo_cls.load(model_path, env=tmp_env)
        tmp_env.close()

        # model already has VecNormalize stats loaded

    # ── Run multi-seed evaluation ──
    print("-" * 60)
    results = []
    for seed in args.seeds:
        metrics = eval_single_seed(
            model=model,
            vecnormalize_path=vecnorm_path,
            obs_mean=obs_mean if args.baseline else None,
            obs_var=obs_var if args.baseline else None,
            n_eval_episodes=args.n_eval_episodes,
            base_seed=seed,
        )
        results.append(metrics)

        print(f"[seed {seed:4d}] "
              f"success={metrics['success_rate']:.3f} "
              f"({metrics['success_count']}/{args.n_eval_episodes}) | "
              f"return={metrics['mean_return']:.1f}±{metrics['std_return']:.1f} | "
              f"len={metrics['mean_length']:.0f}±{metrics['std_length']:.0f} | "
              f"smooth={metrics['mean_action_smoothness']:.1f} | "
              f"min_dist={metrics['mean_min_hand_dist']:.4f}")

    # ── Aggregate ──
    success_rates = [r["success_rate"] for r in results]
    returns = [r["mean_return"] for r in results]
    lengths = [r["mean_length"] for r in results]
    smoothness = [r["mean_action_smoothness"] for r in results]
    min_dists = [r["mean_min_hand_dist"] for r in results]

    print("-" * 60)
    print(f"Aggregate over {len(args.seeds)} seeds:")
    print(f"  Success rate       : {np.mean(success_rates):.3f} "
          f"± {np.std(success_rates):.3f}")
    print(f"  Mean return        : {np.mean(returns):.1f} "
          f"± {np.std(returns):.1f}")
    print(f"  Mean length        : {np.mean(lengths):.0f} "
          f"± {np.std(lengths):.0f}")
    print(f"  Action smoothness  : {np.mean(smoothness):.1f} "
          f"± {np.std(smoothness):.1f}")
    print(f"  Min hand distance  : {np.mean(min_dists):.4f} "
          f"± {np.std(min_dists):.4f}")

    return results


if __name__ == "__main__":
    args = parse_args()
    eval_multi_seed(args)
