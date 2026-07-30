"""
Minimal fine-tuning script for PPO vs GRPO comparison on raw HumanoidBench door task.

No curriculum wrappers, no custom reward shaping — uses the base environment's reward.
Loads pretrained model from curriculum-v7.1 checkpoint.

Usage:
    # PPO fine-tune
    python finetune_hb.py --algo ppo --env_name h1hand-door-v0 --exp_name ft_ppo_v1 --seed 42

    # GRPO fine-tune
    python finetune_hb.py --algo grpo --env_name h1hand-door-v0 --exp_name ft_grpo_v1 --seed 42

Metrics collected:
    - Sample Efficiency: env steps vs reward/success (via tensorboard/W&B)
    - Final Performance: success rate + cumulative reward (via evaluation)
    - Training Stability: variance of per-episode returns (via EpisodeLogCallback)
    - Behavior Quality: action smoothness (via ActionSmoothnessCallback)
"""

import argparse
import os
from collections import deque

import numpy as np
import gymnasium as gym
from gymnasium.wrappers import TimeLimit

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.running_mean_std import RunningMeanStd

import wandb
import humanoid_bench

from utils.grpo import GRPO

# ────────────────────────────────────────────────────────────────────────────
# CLI
# ────────────────────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser()
parser.add_argument("--algo", type=str, default="ppo", choices=["ppo", "grpo"])
parser.add_argument("--env_name", type=str, default="h1hand-door-v0")
parser.add_argument("--exp_name", type=str, required=True)
parser.add_argument("--seed", type=int, required=True)
parser.add_argument("--num_envs", default=4, type=int)
parser.add_argument("--learning_rate", default=3e-5, type=float)
parser.add_argument("--max_steps", default=5_000_000, type=int)
parser.add_argument("--eval_freq", default=100_000, type=int)
parser.add_argument("--n_eval_episodes", default=10, type=int)
parser.add_argument("--pretrained_path", type=str,
                    default="models/h1hand-door-v0/curriculum-v7.1")
parser.add_argument("--phase2", action="store_true",
                    help="Phase 2: Oracle eval — replace env reward with binary GT success during evaluation")
parser.add_argument("--wandb_entity", default="wlx-k-s-2003-ucl", type=str)
ARGS = parser.parse_args()


# ────────────────────────────────────────────────────────────────────────────
# Minimal wrapper: tracks success & action smoothness ONLY (no reward change)
# ────────────────────────────────────────────────────────────────────────────

class EvalMetricsWrapper(gym.Wrapper):
    """Tracks success & action smoothness.  Optionally replaces the reward with
    a binary oracle signal during evaluation (Phase 2)."""

    def __init__(self, env, oracle_reward: bool = False,
                 oracle_x_threshold: float | None = None):
        super().__init__(env)
        self.oracle_reward = oracle_reward
        self.oracle_x_threshold = oracle_x_threshold  # None → use _episode_success
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

        # success tracking (ground-truth: robot past door-frame)
        if m["robot_x"] > 0.8:
            self._episode_success = True

        # action smoothness (cumulative ||a_t - a_{t-1}||^2)
        if self._prev_action is not None:
            self._episode_action_sqdiff += float(
                np.sum(np.square(action - self._prev_action))
            )
        self._prev_action = action.copy()

        # Phase 2: replace reward with sparse binary oracle
        if self.oracle_reward:
            # Determine oracle condition (separate from eval success tracking)
            if self.oracle_x_threshold is not None:
                oracle_success = m["robot_x"] > self.oracle_x_threshold
            else:
                oracle_success = self._episode_success
            # extremely sparse: 1 at terminal step if oracle condition met, else 0
            if terminated or truncated:
                reward = 1.0 if oracle_success else 0.0
            else:
                reward = 0.0

        # report at episode end
        if terminated or truncated:
            if self._episode_success:
                info["success"] = 1
            info["action_smoothness"] = self._episode_action_sqdiff

        return obs, reward, terminated, truncated, info


# ────────────────────────────────────────────────────────────────────────────
# Environment factory
# ────────────────────────────────────────────────────────────────────────────

def make_env(rank, seed=0, oracle_reward: bool = False,
             oracle_x_threshold: float | None = None):
    def _init():
        env = gym.make(ARGS.env_name)
        env = EvalMetricsWrapper(env, oracle_reward=oracle_reward,
                                 oracle_x_threshold=oracle_x_threshold)
        env = TimeLimit(env, max_episode_steps=1000)
        env = Monitor(env, f"logs/{ARGS.env_name}/{ARGS.exp_name}/train/env_{rank}")
        env.reset(seed=ARGS.seed + rank)
        return env
    return _init


# ────────────────────────────────────────────────────────────────────────────
# Callbacks
# ────────────────────────────────────────────────────────────────────────────

class EpisodeLogCallback(BaseCallback):
    """Log per-episode return, length, and success rate."""

    def __init__(self):
        super().__init__()
        self.ep_returns = []
        self.ep_lengths = []
        self.ep_successes = []
        self.ep_action_smoothness = []

    def _on_step(self) -> bool:
        infos = self.locals["infos"]
        for idx, info in enumerate(infos):
            if "episode" in info:
                self.ep_returns.append(info["episode"]["r"])
                self.ep_lengths.append(info["episode"]["l"])
                self.ep_successes.append(info.get("success", 0))
                self.ep_action_smoothness.append(
                    info.get("action_smoothness", 0.0)
                )
        return True

    def _on_rollout_end(self) -> None:
        if self.ep_returns:
            self.logger.record("results/mean_return", np.mean(self.ep_returns))
            self.logger.record("results/std_return", np.std(self.ep_returns))
            self.logger.record("results/mean_length", np.mean(self.ep_lengths))
            self.logger.record("results/success_rate", np.mean(self.ep_successes))
            # Training stability proxy: std of returns
            # self.logger.record("results/return_std", np.std(self.ep_returns))
        if self.ep_action_smoothness:
            # Behavior quality: mean per-episode action smoothness
            self.logger.record("behavior/action_smoothness",
                               np.mean(self.ep_action_smoothness))
        self.ep_returns = []
        self.ep_lengths = []
        self.ep_successes = []
        self.ep_action_smoothness = []


class EvalBestModelHandler(BaseCallback):
    """Sync VecNormalize stats to eval env when a new best eval model is found."""

    def __init__(self, eval_env):
        super().__init__()
        self.eval_env = eval_env

    def _on_step(self) -> bool:
        train_env = self.model.get_vec_normalize_env()
        if train_env is not None:
            self.eval_env.obs_rms = train_env.obs_rms
            self.eval_env.ret_rms = train_env.ret_rms
        return True


class PeriodicEvalCallback(EvalCallback):
    """EvalCallback that:
    - Uses **dual criterion** for best-model: best-by-success + best-by-reward
    - Saves periodic snapshots at every eval checkpoint
    """

    def __init__(self, model_save_dir: str, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model_save_dir = model_save_dir
        self.best_eval_success_rate = -1.0
        self.best_eval_mean_reward = -float("inf")

    def _on_step(self) -> bool:
        if self.eval_freq <= 0 or self.n_calls % self.eval_freq != 0:
            return super()._on_step()

        # ── Sync VecNormalize before eval ──
        train_vec_norm = self.model.get_vec_normalize_env()
        if train_vec_norm is not None and hasattr(self.eval_env, "obs_rms"):
            self.eval_env.obs_rms = train_vec_norm.obs_rms
            self.eval_env.ret_rms = train_vec_norm.ret_rms

        # ── Run evaluation with success tracking ──
        n_episodes = self.n_eval_episodes
        episode_rewards = []       # reward the agent sees (oracle in phase2)
        episode_successes = []
        episode_lengths = []

        for _ in range(n_episodes):
            obs = self.eval_env.reset()
            done = False
            ep_rew = 0.0
            ep_len = 0
            ep_success = False

            while not done:
                action, _ = self.model.predict(obs, deterministic=self.deterministic)
                obs, rewards, dones, infos = self.eval_env.step(action)
                ep_rew += float(rewards[0])
                ep_len += 1
                if infos[0].get("success", 0) == 1:
                    ep_success = True
                done = bool(dones[0])

            episode_rewards.append(ep_rew)
            episode_successes.append(1.0 if ep_success else 0.0)
            episode_lengths.append(ep_len)

        mean_reward = float(np.mean(episode_rewards))
        std_reward = float(np.std(episode_rewards))
        success_rate = float(np.mean(episode_successes))
        mean_length = float(np.mean(episode_lengths))
        criterion_reward = mean_reward  # best-by-reward uses the reward signal the agent sees

        # ── Log ──
        self.logger.record("eval/mean_reward", mean_reward)
        self.logger.record("eval/std_reward", std_reward)
        self.logger.record("eval/success_rate", success_rate)
        self.logger.record("eval/mean_ep_length", mean_length)

        # Write eval log to disk (mirrors parent class behaviour)
        if self.log_path is not None:
            self.evaluations_timesteps.append(self.num_timesteps)
            self.evaluations_results.append([mean_reward, std_reward])
            self.evaluations_length.append(mean_length)
            np.save(self.log_path, {
                "timesteps": self.evaluations_timesteps,
                "results": self.evaluations_results,
                "ep_lengths": self.evaluations_length,
            })

        self.last_mean_reward = mean_reward
        self.update_child_locals(dict(
            mean_reward=mean_reward,
            std_reward=std_reward,
            success_rate=success_rate,
            mean_length=mean_length,
        ))

        # ── Dual-criterion best model ──
        new_best = False

        # Criterion 1: best by success rate
        if success_rate >= self.best_eval_success_rate:
            self.best_eval_success_rate = success_rate
            self._save_best("best_by_success", train_vec_norm)
            new_best = True

        # Criterion 2: best by (original) mean reward
        if criterion_reward >= self.best_eval_mean_reward:
            self.best_eval_mean_reward = criterion_reward
            self._save_best("best_by_reward", train_vec_norm)
            new_best = True

        # Trigger on_new_best callbacks (VecNormalize sync etc.)
        if new_best and self.best_model_save_path is not None:
            cbs = self.callback_on_new_best
            if cbs is not None:
                for cb in (cbs if isinstance(cbs, (list, tuple)) else [cbs]):
                    cb.update_locals(self.locals)
                    cb.on_step()

        # ── Periodic snapshot ──
        steps = self.num_timesteps
        self.model.save(os.path.join(self.model_save_dir, f"model_{steps}_steps.zip"))
        if train_vec_norm is not None:
            train_vec_norm.save(os.path.join(self.model_save_dir, f"vecnormalize_{steps}_steps.pkl"))

        return True

    def _save_best(self, tag: str, train_vec_norm):
        """Save model + vecnormalize under a given tag prefix."""
        if self.best_model_save_path is None:
            return
        os.makedirs(self.best_model_save_path, exist_ok=True)
        self.model.save(os.path.join(self.best_model_save_path, f"{tag}.zip"))
        if train_vec_norm is not None:
            train_vec_norm.save(os.path.join(self.best_model_save_path, f"{tag}_vecnormalize.pkl"))


# ────────────────────────────────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────────────────────────────────

def main():
    # ── Training env ──
    # Phase 2: train with sparse oracle reward (robot_x > 0.8)
    train_oracle = ARGS.phase2
    train_x_threshold = 0.8 if ARGS.phase2 else None
    env = SubprocVecEnv([
        make_env(i, oracle_reward=train_oracle, oracle_x_threshold=train_x_threshold)
        for i in range(ARGS.num_envs)
    ])
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0)

    # ── Eval env ──
    EVAL_SEED_OFFSET = 1000
    model_save_path = f"models/{ARGS.env_name}/{ARGS.exp_name}"
    os.makedirs(model_save_path, exist_ok=True)

    eval_env = DummyVecEnv([make_env(EVAL_SEED_OFFSET, oracle_reward=ARGS.phase2)])
    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False, clip_obs=10.0, training=False)

    # ── Load pretrained model + VecNormalize stats ──
    pretrained_model = os.path.join(ARGS.pretrained_path, "best_model.zip")
    pretrained_vecnorm = os.path.join(ARGS.pretrained_path, "best_vecnormalize.pkl")

    algo_cls = {"ppo": PPO, "grpo": GRPO}[ARGS.algo]
    print(f"Algorithm : {ARGS.algo}")
    print(f"Pretrained: {pretrained_model}")

    model = algo_cls.load(
        pretrained_model,
        env=env,
        tensorboard_log=f"runs/baseline_{ARGS.env_name}_{ARGS.exp_name}",
    )

    # Load obs normalization, reset reward stats (raw env has different reward)
    env = VecNormalize.load(pretrained_vecnorm, env.venv)
    env.training = True
    env.ret_rms = RunningMeanStd(shape=())
    env.returns = np.zeros(ARGS.num_envs)

    eval_env.obs_rms = env.obs_rms
    eval_env.ret_rms = env.ret_rms
    eval_env.training = False

    model.set_env(env)

    # ── Callbacks ──
    eval_callback = PeriodicEvalCallback(
        model_save_dir=model_save_path,
        eval_env=eval_env,
        best_model_save_path=model_save_path,
        callback_on_new_best=EvalBestModelHandler(
            eval_env=eval_env,
        ),
        log_path=f"logs/{ARGS.env_name}/{ARGS.exp_name}/eval",
        eval_freq=ARGS.eval_freq // ARGS.num_envs,
        n_eval_episodes=ARGS.n_eval_episodes,
        deterministic=True,
        render=False,
        verbose=1,
    )

    run = wandb.init(
        entity=ARGS.wandb_entity,
        project="humanoid-bench",
        name=f"{ARGS.exp_name}",
        tags=[f"algo_{ARGS.algo}", f"seed_{ARGS.seed}", "finetune_raw"]
             + (["phase2"] if ARGS.phase2 else []),
        sync_tensorboard=True,
        monitor_gym=True,
        save_code=False,
        config=vars(ARGS),
    )

    # ── Train ──
    try:
        model.learn(
            total_timesteps=ARGS.max_steps,
            log_interval=1,
            reset_num_timesteps=False,
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
    env.close()


if __name__ == "__main__":
    main()
