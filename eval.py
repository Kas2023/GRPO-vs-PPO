"""
Multi-seed evaluation on raw HumanoidBench door task (no curriculum wrappers).

Captures the same metrics as finetune_hb.py:
    - Success rate (door_openness > 0.4 and robot_x > door_x)
    - Cumulative reward
    - Episode length
    - Action smoothness (cumulative ||a_t - a_{t-1}||^2)

Usage:
    python eval.py \
        --exp_name curriculum-v7.1 \
        --seeds 42 2024 777 88 13 \
        --n_eval_episodes 20
"""
import os
import argparse
import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
import humanoid_bench

from utils.grpo import GRPO


# ────────────────────────────────────────────────────────────────────────────
# Thin wrapper (same as finetune_hb.py) — no reward modification
# ────────────────────────────────────────────────────────────────────────────

class EvalMetricsWrapper(gym.Wrapper):
    """Tracks success & action smoothness. Does NOT modify the reward."""

    def __init__(self, env):
        super().__init__(env)
        self._episode_success = False
        self._prev_action = None
        self._episode_action_sqdiff = 0.0

    def _get_metrics(self):
        task = self.env.unwrapped.task
        left_dist = np.linalg.norm(
            task._env.data.body("door_hatch").xpos
            - task._env.named.data.site_xpos["left_hand"]
        )
        right_dist = np.linalg.norm(
            task._env.data.body("door_hatch").xpos
            - task._env.named.data.site_xpos["right_hand"]
        )
        door_pos = task._env.data.body("door").xpos
        return {
            "hand_distance": min(left_dist, right_dist),
            "hatch_angle": task._env.data.qpos[-1],
            "door_openness": task._env.data.qpos[-2],
            "robot_x": task._env.named.data.site_xpos["imu", "x"],
            "door_x": door_pos[0],
        }

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._episode_success = False
        self._prev_action = None
        self._episode_action_sqdiff = 0.0
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        m = self._get_metrics()

        if m["door_openness"] > 0.4 and m["robot_x"] > m["door_x"]:
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

        return obs, reward, terminated, truncated, info


# ────────────────────────────────────────────────────────────────────────────
# CLI
# ────────────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_name", type=str, default="h1hand-door-v0")
    parser.add_argument("--exp_name", type=str, required=True)
    parser.add_argument("--model_dir", type=str, default="models")
    parser.add_argument("--model_file", type=str, default="best_model.zip")
    parser.add_argument("--vecnormalize_file", type=str, default="best_vecnormalize.pkl")
    parser.add_argument("--algo", type=str, default="ppo", choices=["ppo", "grpo"])
    parser.add_argument("--seeds", type=int, nargs="+", default=[1024, 2024, 777, 88, 13])
    parser.add_argument("--n_eval_episodes", type=int, default=20)
    return parser.parse_args()


# ────────────────────────────────────────────────────────────────────────────
# Single-seed evaluation
# ────────────────────────────────────────────────────────────────────────────

def eval_single_seed(
    env_name: str,
    seed: int,
    model_path: str,
    vecnormalize_path: str,
    algo: str = "ppo",
    n_eval_episodes: int = 20,
):
    # Build raw env with EvalMetricsWrapper (no curriculum wrappers)
    env = gym.make(env_name)
    env = EvalMetricsWrapper(env)
    env.reset(seed=seed)
    env = DummyVecEnv([lambda: env])

    # Load model
    algo_cls = {"ppo": PPO, "grpo": GRPO}[algo]
    model = algo_cls.load(model_path, env=env)

    # Load VecNormalize obs stats (eval doesn't need reward stats)
    if os.path.exists(vecnormalize_path):
        env = VecNormalize.load(vecnormalize_path, env)
        env.training = False
        env.norm_reward = False
    else:
        print(f"[WARN] VecNormalize file not found: {vecnormalize_path}")

    success_count = 0
    episode_returns = []
    episode_lengths = []
    episode_action_smoothness = []

    for ep in range(n_eval_episodes):
        # Re-seed before each episode for independent rollouts
        env.venv.envs[0].reset(seed=seed + ep)
        obs = env.reset()
        done = False
        ep_return = 0.0
        ep_length = 0
        ep_success = False

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, dones, info = env.step(action)
            ep_return += reward[0]
            ep_length += 1

            env_info = info[0]
            if env_info.get("success", 0) == 1:
                ep_success = True

            done = dones[0]

        if ep_success:
            success_count += 1
        episode_returns.append(ep_return)
        episode_lengths.append(ep_length)
        episode_action_smoothness.append(
            info[0].get("action_smoothness", 0.0)
        )

    env.close()

    return {
        "seed": seed,
        "success_rate": success_count / n_eval_episodes,
        "success_count": success_count,
        "mean_return": float(np.mean(episode_returns)),
        "std_return": float(np.std(episode_returns)),
        "mean_length": float(np.mean(episode_lengths)),
        "std_length": float(np.std(episode_lengths)),
        "mean_action_smoothness": float(np.mean(episode_action_smoothness)),
    }


# ────────────────────────────────────────────────────────────────────────────
# Multi-seed evaluation
# ────────────────────────────────────────────────────────────────────────────

def eval_multi_seed(
    env_name: str,
    exp_name: str,
    model_dir: str = "models",
    model_file: str = "best_model.zip",
    vecnormalize_file: str = "best_vecnormalize.pkl",
    algo: str = "ppo",
    seeds: list = None,
    n_eval_episodes: int = 20,
):
    if seeds is None:
        seeds = [1024, 2024, 777, 88, 13]

    model_path = os.path.join(model_dir, env_name, exp_name, model_file)
    vecnormalize_path = os.path.join(model_dir, env_name, exp_name, vecnormalize_file)

    print(f"Model      : {model_path}")
    print(f"VecNorm    : {vecnormalize_path}")
    print(f"Algorithm  : {algo}")
    print(f"Seeds      : {seeds}")
    print(f"Episodes   : {n_eval_episodes} per seed")
    print("-" * 50)

    results = []
    for seed in seeds:
        metrics = eval_single_seed(
            env_name=env_name,
            seed=seed,
            model_path=model_path,
            vecnormalize_path=vecnormalize_path,
            algo=algo,
            n_eval_episodes=n_eval_episodes,
        )
        results.append(metrics)

        print(f"[seed {seed:4d}] "
              f"success_rate={metrics['success_rate']:.3f} "
              f"({metrics['success_count']}/{n_eval_episodes}) | "
              f"return={metrics['mean_return']:.1f} +/- {metrics['std_return']:.1f} | "
              f"length={metrics['mean_length']:.0f} +/- {metrics['std_length']:.0f} | "
              f"action_smooth={metrics['mean_action_smoothness']:.1f}")

    # ── Aggregate ──
    success_rates = [r["success_rate"] for r in results]
    returns = [r["mean_return"] for r in results]
    action_smoothness = [r["mean_action_smoothness"] for r in results]

    print("-" * 50)
    print(f"Aggregate over {len(seeds)} seeds:")
    print(f"  Success rate       : {np.mean(success_rates):.3f} +/- {np.std(success_rates):.3f}")
    print(f"  Mean return        : {np.mean(returns):.1f} +/- {np.std(returns):.1f}")
    print(f"  Action smoothness  : {np.mean(action_smoothness):.1f} +/- {np.std(action_smoothness):.1f}")

    return results


if __name__ == "__main__":
    args = parse_args()
    eval_multi_seed(
        env_name=args.env_name,
        exp_name=args.exp_name,
        model_dir=args.model_dir,
        model_file=args.model_file,
        vecnormalize_file=args.vecnormalize_file,
        algo=args.algo,
        seeds=args.seeds,
        n_eval_episodes=args.n_eval_episodes,
    )
