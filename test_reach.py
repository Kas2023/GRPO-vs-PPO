"""
Demo: Direct Fine-tuning of Pretrained Reach Model on h1hand-reach-v0
======================================================================

Part 5 实验方向 — 利用 humanoid-bench/data/reach_one_hand 预训练参数，
在 reach 任务本身上进行 PPO vs GRPO 微调对比。

与 Door 任务不同，reach 任务具有更清晰的奖励结构:
  - healthy_reward: 躯干直立程度
  - motion_penalty: 关节速度惩罚
  - reward_close: 手距目标 < 1m 时 +5
  - reward_success: 手距目标 < 0.05m 时 +10

这提供了一个更"干净"的环境来对比 PPO vs GRPO 的微调行为。

核心设计:
  1. 从 torch_model.pt 加载预训练权重到 SB3 policy 网络
  2. 从 mean.npy / var.npy 初始化 VecNormalize 统计量
  3. 用 ReachObsWrapper 将 155D 观测映射为模型期望的 55D
  4. 用 ReachActWrapper 将 19D 身体动作扩展为 61D（手部固定）
  5. PPO vs GRPO 微调对比

用法:
  python test_reach.py --algo ppo  --exp_name reach_ppo  --seed 42
  python test_reach.py --algo grpo --exp_name reach_grpo --seed 42
"""

import argparse
import os
import sys
import numpy as np
import torch as th
import gymnasium as gym
from gymnasium.wrappers import TimeLimit
from gymnasium.spaces import Box

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.running_mean_std import RunningMeanStd

import mujoco
import humanoid_bench
from humanoid_bench.wrappers import get_body_idxs
from utils.grpo import GRPO

# ──────────────────────────────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────────────────────────────
REACH_MODEL_DIR = "humanoid-bench/data/reach_one_hand"
REACH_MODEL_PATH = os.path.join(REACH_MODEL_DIR, "torch_model.pt")
REACH_MEAN_PATH = os.path.join(REACH_MODEL_DIR, "mean.npy")
REACH_VAR_PATH = os.path.join(REACH_MODEL_DIR, "var.npy")

# ──────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--algo", type=str, default="ppo", choices=["ppo", "grpo"])
parser.add_argument("--exp_name", type=str, required=True)
parser.add_argument("--seed", type=int, required=True)
parser.add_argument("--num_envs", default=4, type=int)
parser.add_argument("--learning_rate", default=3e-5, type=float)
parser.add_argument("--max_steps", default=2_000_000, type=int)
parser.add_argument("--eval_freq", default=50_000, type=int)
parser.add_argument("--n_eval_episodes", default=20, type=int)
parser.add_argument("--wandb_entity", default="wlx-k-s-2003-ucl", type=str)
ARGS = parser.parse_args()

# ──────────────────────────────────────────────────────────────────────
# Observation wrapper: 155D → 55D (reach model format)
# ──────────────────────────────────────────────────────────────────────

class ReachObsWrapper(gym.ObservationWrapper):
    """Transform full 155D reach-task observation into the 55D format
    expected by the pretrained reach model.

    Full obs (155D):
        position(75) + velocity(74) + left_hand(3) + target(3)

    Reach-model obs (55D):
        position[body_idxs][2:] + velocity[body_vel_idxs] + left_hand + target

    The x,y offset subtraction (used in SingleReachWrapper.get_reach_obs)
    centres the observation on the robot's current position.
    """

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
        # Split the 155D observation
        position = obs[:robot_dof]                        # 75D
        velocity = obs[robot_dof:robot_dof * 2 - 1]       # 74D
        left_hand = obs[robot_dof * 2 - 1:robot_dof * 2 + 2]  # 3D
        target = obs[robot_dof * 2 + 2:]                  # 3D

        # Extract body-only components
        body_pos = position[self.body_idxs]               # 26D
        body_vel = velocity[self.body_vel_idxs]           # 25D

        # Subtract (x, y, 0) offset to centre on robot
        offset = np.array([body_pos[0], body_pos[1], 0.0])
        body_pos[:3] = body_pos[:3] - offset
        left_hand = left_hand - offset
        target = target - offset

        # Build 55D: position[2:] + velocity + left_hand + target
        # position[2:] = 24D, velocity = 25D, left_hand = 3D, target = 3D
        return np.concatenate([body_pos[2:], body_vel, left_hand, target])


# ──────────────────────────────────────────────────────────────────────
# Action wrapper: 19D → 61D (expand body actions, fix hands)
# ──────────────────────────────────────────────────────────────────────

class ReachActWrapper(gym.ActionWrapper):
    """Expand 19D body-joint actions to 61D (body + hands).

    The reach model was trained on body joints only.  Hand joints are
    fixed at a neutral position (1.57 rad, the default used in
    SingleReachWrapper).
    """

    def __init__(self, env):
        super().__init__(env)
        self.action_space = Box(low=-1, high=1, shape=(19,), dtype=np.float32)

        # Determine which indices correspond to body vs hand joints
        if env.unwrapped.model.nu > 19:
            self.act_idxs = list(range(15)) + list(range(16, 20))  # 19 body joints
        else:
            self.act_idxs = list(range(19))

        self._full_nu = env.unwrapped.model.nu

    def action(self, action: np.ndarray) -> np.ndarray:
        if self._full_nu <= 19:
            return action

        # Unnormalize from [-1, 1] to actual joint range for body joints
        env = self.env.unwrapped
        body_action = (action + 1) / 2 * (
            env.action_high[self.act_idxs] - env.action_low[self.act_idxs]
        ) + env.action_low[self.act_idxs]

        # Build full action: body + fixed hands
        full_action = env.data.ctrl.copy()
        full_action[self.act_idxs] = body_action
        full_action[15] = 1.57   # left hand
        full_action[20] = 1.57   # right hand

        # Normalize back to [-1, 1]
        return 2 * (full_action - env.action_low) / (env.action_high - env.action_low) - 1


# ──────────────────────────────────────────────────────────────────────
# Metrics wrapper for reach task
# ──────────────────────────────────────────────────────────────────────

class ReachMetricsWrapper(gym.Wrapper):
    """Tracks hand distance, success (hand_dist < 0.05), return,
    and action smoothness for the reach task."""

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

        # Track hand distance (from env reward info)
        # The Reach.get_reward() returns reward_info with "hand_dist"
        # Monitor puts per-step reward info into info dict
        hand_dist = info.get("hand_dist", None)
        if hand_dist is not None and hand_dist < self._min_hand_dist:
            self._min_hand_dist = hand_dist
        if hand_dist is not None and hand_dist < 0.05:
            self._episode_success = True

        # Action smoothness on the 19D body actions
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

def make_env(rank: int, seed: int = 0):
    def _init():
        env = gym.make("h1hand-reach-v0", render_mode="rgb_array")
        env = ReachObsWrapper(env)       # 155D → 55D observation
        env = ReachActWrapper(env)       # 19D → 61D action
        env = ReachMetricsWrapper(env)   # metrics tracking
        env = TimeLimit(env, max_episode_steps=1000)
        env = Monitor(env, f"logs/h1hand-reach-v0/{ARGS.exp_name}/train/env_{rank}")
        env.reset(seed=ARGS.seed + rank)
        return env
    return _init


# ──────────────────────────────────────────────────────────────────────
# Callbacks
# ──────────────────────────────────────────────────────────────────────

class EpisodeLogCallback(BaseCallback):
    def __init__(self):
        super().__init__()
        self.ep_returns = []
        self.ep_lengths = []
        self.ep_successes = []
        self.ep_action_smoothness = []
        self.ep_min_hand_dist = []

    def _on_step(self) -> bool:
        infos = self.locals["infos"]
        for info in infos:
            if "episode" in info:
                self.ep_returns.append(info["episode"]["r"])
                self.ep_lengths.append(info["episode"]["l"])
                self.ep_successes.append(info.get("success", 0))
                self.ep_action_smoothness.append(
                    info.get("action_smoothness", 0.0))
                self.ep_min_hand_dist.append(
                    info.get("min_hand_dist", float("inf")))
        return True

    def _on_rollout_end(self) -> None:
        if self.ep_returns:
            self.logger.record("results/mean_return", np.mean(self.ep_returns))
            self.logger.record("results/mean_length", np.mean(self.ep_lengths))
            self.logger.record("results/success_rate", np.mean(self.ep_successes))
        if self.ep_action_smoothness:
            self.logger.record("behavior/action_smoothness",
                               np.mean(self.ep_action_smoothness))
        if self.ep_min_hand_dist:
            self.logger.record("results/min_hand_dist",
                               np.mean(self.ep_min_hand_dist))
        self.ep_returns = []
        self.ep_lengths = []
        self.ep_successes = []
        self.ep_action_smoothness = []
        self.ep_min_hand_dist = []


class PeriodicEvalCallback(BaseCallback):
    def __init__(self, eval_env, model_save_dir: str,
                 eval_freq: int, n_eval_episodes: int):
        super().__init__()
        self.eval_env = eval_env
        self.model_save_dir = model_save_dir
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes
        self.best_eval_success_rate = -1.0
        self.best_eval_mean_reward = -float("inf")

    def _on_step(self) -> bool:
        if self.n_calls % self.eval_freq != 0:
            return True

        # Sync VecNormalize
        train_vec_norm = self.model.get_vec_normalize_env()
        if train_vec_norm is not None:
            self.eval_env.obs_rms = train_vec_norm.obs_rms
            self.eval_env.ret_rms = train_vec_norm.ret_rms

        episode_rewards = []
        episode_successes = []
        episode_lengths = []
        episode_min_hand_dists = []

        for _ in range(self.n_eval_episodes):
            obs = self.eval_env.reset()
            done = False
            ep_rew = 0.0
            ep_len = 0
            ep_success = False
            ep_min_dist = float("inf")

            while not done:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, rewards, dones, infos = self.eval_env.step(action)
                ep_rew += float(rewards[0])
                ep_len += 1
                if infos[0].get("success", 0) == 1:
                    ep_success = True
                d = infos[0].get("min_hand_dist", float("inf"))
                if d < ep_min_dist:
                    ep_min_dist = d
                done = bool(dones[0])

            episode_rewards.append(ep_rew)
            episode_successes.append(1.0 if ep_success else 0.0)
            episode_lengths.append(ep_len)
            episode_min_hand_dists.append(ep_min_dist)

        mean_reward = float(np.mean(episode_rewards))
        success_rate = float(np.mean(episode_successes))
        mean_length = float(np.mean(episode_lengths))
        mean_min_dist = float(np.mean(episode_min_hand_dists))

        self.logger.record("eval/mean_reward", mean_reward)
        self.logger.record("eval/success_rate", success_rate)
        self.logger.record("eval/mean_ep_length", mean_length)
        self.logger.record("eval/min_hand_dist", mean_min_dist)

        steps = self.num_timesteps

        # Save periodic snapshot
        self.model.save(os.path.join(self.model_save_dir, f"model_{steps}_steps"))
        if train_vec_norm is not None:
            train_vec_norm.save(os.path.join(
                self.model_save_dir, f"vecnormalize_{steps}_steps.pkl"))

        # Best by success
        if success_rate >= self.best_eval_success_rate:
            self.best_eval_success_rate = success_rate
            self.model.save(os.path.join(self.model_save_dir, "best_by_success"))
            if train_vec_norm is not None:
                train_vec_norm.save(os.path.join(
                    self.model_save_dir, "best_by_success_vecnormalize.pkl"))

        # Best by reward
        if mean_reward >= self.best_eval_mean_reward:
            self.best_eval_mean_reward = mean_reward
            self.model.save(os.path.join(self.model_save_dir, "best_by_reward"))
            if train_vec_norm is not None:
                train_vec_norm.save(os.path.join(
                    self.model_save_dir, "best_by_reward_vecnormalize.pkl"))

        return True


# ──────────────────────────────────────────────────────────────────────
# Weight loading: TorchModel → SB3 policy
# ──────────────────────────────────────────────────────────────────────

def load_reach_weights_to_policy(model, pt_path: str, mean_path: str, var_path: str):
    """Load pretrained reach TorchModel weights into an SB3 policy network.

    Architecture mapping:
        TorchModel.dense1 → mlp_extractor.policy_net.0  (55→256, Tanh)
        TorchModel.dense2 → mlp_extractor.policy_net.2  (256→256, Tanh)
        TorchModel.dense3 → action_net                   (256→19)

    Value network and log_std remain randomly initialized.
    """
    pt_weights = th.load(pt_path, map_location="cpu")

    policy_state = model.policy.state_dict()

    # Map weights
    mapping = {
        "dense1.weight": "mlp_extractor.policy_net.0.weight",
        "dense1.bias":   "mlp_extractor.policy_net.0.bias",
        "dense2.weight": "mlp_extractor.policy_net.2.weight",
        "dense2.bias":   "mlp_extractor.policy_net.2.bias",
        "dense3.weight": "action_net.weight",
        "dense3.bias":   "action_net.bias",
    }

    loaded = 0
    for pt_key, sb3_key in mapping.items():
        if pt_key in pt_weights and sb3_key in policy_state:
            if pt_weights[pt_key].shape == policy_state[sb3_key].shape:
                policy_state[sb3_key] = pt_weights[pt_key]
                loaded += 1
            else:
                print(f"  [WARN] Shape mismatch: {pt_key} "
                      f"{pt_weights[pt_key].shape} vs {sb3_key} "
                      f"{policy_state[sb3_key].shape}")
        else:
            print(f"  [WARN] Key not found: {pt_key} or {sb3_key}")

    model.policy.load_state_dict(policy_state)
    print(f"  Loaded {loaded}/6 weight tensors into policy network")

    # Set up VecNormalize observation stats
    pt_mean = np.load(mean_path)[0].astype(np.float64)  # (55,) — first row
    pt_var = np.load(var_path)[0].astype(np.float64)    # (55,)

    train_env = model.get_vec_normalize_env()
    if train_env is not None:
        train_env.obs_rms.mean = pt_mean.copy()
        train_env.obs_rms.var = pt_var.copy()
        train_env.obs_rms.count = 1e4  # high count → slow update
        print(f"  Initialized VecNormalize obs stats from reach data")


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────

def main():
    # ── Validate paths ──
    for p in [REACH_MODEL_PATH, REACH_MEAN_PATH, REACH_VAR_PATH]:
        if not os.path.exists(p):
            print(f"[ERROR] Missing: {p}")
            sys.exit(1)

    env_name = "h1hand-reach-v0"
    model_save_path = f"models/{env_name}/{ARGS.exp_name}"
    os.makedirs(model_save_path, exist_ok=True)

    # ── Training envs ──
    env = SubprocVecEnv([make_env(i) for i in range(ARGS.num_envs)])
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0)

    # ── Eval env ──
    EVAL_SEED_OFFSET = 1000
    eval_env = DummyVecEnv([make_env(EVAL_SEED_OFFSET)])
    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False,
                            clip_obs=10.0, training=False)

    # ── Create model ──
    algo_cls = {"ppo": PPO, "grpo": GRPO}[ARGS.algo]

    print(f"Algorithm     : {ARGS.algo}")
    print(f"Reach model   : {REACH_MODEL_PATH}")
    print(f"Obs space     : {env.observation_space}")
    print(f"Action space  : {env.action_space}")

    model = algo_cls(
        "MlpPolicy",
        env,
        verbose=1,
        tensorboard_log=f"runs/{env_name}_{ARGS.exp_name}",
        learning_rate=ARGS.learning_rate,
        n_steps=2048 // ARGS.num_envs,
        batch_size=256,
        policy_kwargs=dict(
            net_arch=dict(pi=[256, 256], vf=[256, 256]),
        ),
        seed=ARGS.seed,
        device="cuda",
    )

    # ── Load pretrained reach weights ──
    print("\nLoading pretrained reach weights into SB3 policy...")
    load_reach_weights_to_policy(
        model, REACH_MODEL_PATH, REACH_MEAN_PATH, REACH_VAR_PATH
    )

    # Sync eval VecNormalize
    train_vec_norm = model.get_vec_normalize_env()
    eval_env.obs_rms = train_vec_norm.obs_rms
    eval_env.ret_rms = RunningMeanStd(shape=())
    eval_env.training = False

    # ── Callbacks ──
    eval_callback = PeriodicEvalCallback(
        eval_env=eval_env,
        model_save_dir=model_save_path,
        eval_freq=ARGS.eval_freq // ARGS.num_envs,
        n_eval_episodes=ARGS.n_eval_episodes,
    )

    try:
        import wandb
        run = wandb.init(
            entity=ARGS.wandb_entity,
            project="humanoid-bench",
            name=f"{ARGS.exp_name}",
            tags=[f"algo_{ARGS.algo}", f"seed_{ARGS.seed}",
                  "reach_finetune", "part5"],
            sync_tensorboard=True,
            monitor_gym=True,
            save_code=False,
            config=vars(ARGS),
        )
    except Exception:
        run = None
        print("[WARN] wandb not available; skipping.")

    # ── Train ──
    print(f"\nStarting fine-tuning ({ARGS.algo.upper()}) on reach task...")
    try:
        model.learn(
            total_timesteps=ARGS.max_steps,
            log_interval=1,
            reset_num_timesteps=True,
            callback=[
                eval_callback,
                EpisodeLogCallback(),
            ],
        )
    except KeyboardInterrupt:
        print("\nUser interrupted. Saving final state...")
    finally:
        model.save(os.path.join(model_save_path, "latest_model"))
        env.save(os.path.join(model_save_path, "latest_vecnormalize.pkl"))
        env.close()

    print(f"Training finished. Models saved to: {model_save_path}")
    eval_env.close()

    # ── Final evaluation ──
    print("\n" + "=" * 60)
    print("Final Evaluation")
    print("=" * 60)

    for tag in ["best_by_success", "best_by_reward", "latest_model"]:
        model_file = os.path.join(model_save_path, f"{tag}.zip")
        vecnorm_file = os.path.join(model_save_path, f"{tag}_vecnormalize.pkl")
        if not os.path.exists(model_file):
            continue

        # Rebuild eval env for each test (VecNormalize is mutated on load)
        eval_env2 = DummyVecEnv([make_env(EVAL_SEED_OFFSET)])
        eval_env2 = VecNormalize(eval_env2, norm_obs=True, norm_reward=False,
                                 clip_obs=10.0, training=False)

        print(f"\n[{tag}]")
        model2 = algo_cls.load(model_file, env=eval_env2)

        if os.path.exists(vecnorm_file):
            eval_env2 = VecNormalize.load(vecnorm_file, eval_env2)
            eval_env2.training = False
            eval_env2.norm_reward = False

        success_count = 0
        all_returns = []
        all_min_dists = []
        n_ep = ARGS.n_eval_episodes * 3

        for ep in range(n_ep):
            obs = eval_env2.reset()
            done = False
            ep_rew = 0.0
            ep_min_dist = float("inf")
            ep_success = False

            while not done:
                action, _ = model2.predict(obs, deterministic=True)
                obs, rewards, dones, infos = eval_env2.step(action)
                ep_rew += float(rewards[0])
                if infos[0].get("success", 0) == 1:
                    ep_success = True
                d = infos[0].get("min_hand_dist", float("inf"))
                if d < ep_min_dist:
                    ep_min_dist = d
                done = bool(dones[0])

            if ep_success:
                success_count += 1
            all_returns.append(ep_rew)
            all_min_dists.append(ep_min_dist)

        print(f"  Success rate   : {success_count / n_ep:.3f} ({success_count}/{n_ep})")
        print(f"  Mean return    : {np.mean(all_returns):.1f} +/- {np.std(all_returns):.1f}")
        print(f"  Mean min dist  : {np.mean(all_min_dists):.3f} +/- {np.std(all_min_dists):.3f}")
        eval_env2.close()

    if run is not None:
        run.finish()


if __name__ == "__main__":
    main()
