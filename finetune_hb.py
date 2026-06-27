import argparse
import os

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecVideoRecorder, VecNormalize

import gymnasium as gym

from gymnasium.wrappers import TimeLimit
import wandb
from wandb.integration.sb3 import WandbCallback

import humanoid_bench
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np
import torch as th

from collections import deque

from utils.wrappers import DoorCurriculumWrapper

parser = argparse.ArgumentParser()
parser.add_argument("--env_name", type=str)
parser.add_argument("--exp_name", type=str)
parser.add_argument("--seed", type=int)
parser.add_argument("--num_envs", default=4, type=int)
parser.add_argument("--learning_rate", default=1e-4, type=float)
parser.add_argument("--max_steps", default=20000000, type=int)
parser.add_argument("--eval_freq", default=100000, type=int)
parser.add_argument("--n_eval_episodes", default=10, type=int)
parser.add_argument("--stage_thresholds", default="650.0,900.0", type=str)
parser.add_argument("--wandb_entity", default="wlx-k-s-2003-ucl", type=str)
ARGS = parser.parse_args()


def make_env(
    rank,
    seed=0
):
    """
    Utility function for multiprocessed env.

    :param rank: (int) index of the subprocess
    :param seed: (int) the inital seed for RNG
    """

    def _init():
        
        env = gym.make(ARGS.env_name)
        env = DoorCurriculumWrapper(env, stage=1)
        env = TimeLimit(env, max_episode_steps=1000)
        env = Monitor(env, f"logs/{ARGS.env_name}/{ARGS.exp_name}/train/env_{rank}")
        env.reset(seed=ARGS.seed + rank)
        
        return env

    return _init


class BestModelHandler(BaseCallback):
    def __init__(self, eval_env, save_path, render_steps: int = 1000, verbose: int = 0):
        super().__init__(verbose=verbose)
        self.eval_env = eval_env
        self.save_path = save_path
        self.render_steps = render_steps

    def _on_step(self) -> bool:
        print(f"Saving stats and recording...")
        
        # A. 保存最新的 VecNormalize 统计量
        train_env = self.model.get_vec_normalize_env()
        if train_env is not None:
            train_env.save(self.save_path)
            # 同步给评估环境以确保录像效果一致
            self.eval_env.obs_rms = train_env.obs_rms
            self.eval_env.ret_rms = train_env.ret_rms

        # B. 录制视频
        video = []
        obs = self.eval_env.reset()
        for _ in range(self.render_steps):
            action, _ = self.model.predict(obs, deterministic=True)
            obs, _, _, _ = self.eval_env.step(action)
            pixels = self.eval_env.render()
            video.append(pixels.transpose(2, 0, 1)) # HWC -> CHW

        video_array = np.stack(video)
        wandb.log({
            "results/best_video": wandb.Video(video_array, fps=50, format="gif"),
        })
        return True


class CustomEvalCallback(EvalCallback):
    def __init__(self, *args, stage_thresholds: list = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.current_stage = 1
        self.stage_thresholds = stage_thresholds or [80.00, 200.00]

    def _on_step(self) -> bool:
        # 1. 执行原有的同步和评估逻辑
        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            # 同步 VecNormalize
            train_vec_norm = self.model.get_vec_normalize_env()
            if train_vec_norm is not None and isinstance(self.eval_env, VecNormalize):
                self.eval_env.obs_rms = train_vec_norm.obs_rms
                self.eval_env.ret_rms = train_vec_norm.ret_rms
            
            # 2. 调用父类评估 (会更新 self.last_mean_reward)
            continue_training = super()._on_step()

            # 3. 检查是否需要切换课程阶段
            if self.current_stage <= len(self.stage_thresholds):
                if self.last_mean_reward >= self.stage_thresholds[self.current_stage - 1]:
                    self.current_stage += 1
                    print(f"\n[Curriculum] Success! Reached stage {self.current_stage} thresholds.")
                    
                    # 关键：更新训练环境（多进程环境需要用 env_method）
                    self.training_env.env_method("set_stage", self.current_stage)
                    # 关键：更新评估环境
                    self.eval_env.env_method("set_stage", self.current_stage)
            
            return continue_training
        return True


class LogCallback(BaseCallback):
    """
    Custom callback for plotting additional values in tensorboard.
    """

    def __init__(self, verbose=0, info_keywords=()):
        super().__init__(verbose)
        self.aux_rewards = {}
        self.aux_returns = {}
        for key in info_keywords:
            self.aux_rewards[key] = np.zeros(ARGS.num_envs)
            self.aux_returns[key] = deque(maxlen=100)

    def _on_step(self) -> bool:
        infos = self.locals["infos"]
        for idx in range(len(infos)):
            for key in self.aux_rewards.keys():
                self.aux_rewards[key][idx] += infos[idx][key]

            if self.locals['dones'][idx]:
                for key in self.aux_rewards.keys():
                    self.aux_returns[key].append(self.aux_rewards[key][idx])
                    self.aux_rewards[key][idx] = 0
        return True

    def _on_rollout_end(self) -> None:
        
        for key in self.aux_returns.keys():
            self.logger.record("aux_returns_{}/mean".format(key), np.mean(self.aux_returns[key]))


class EpisodeLogCallback(BaseCallback):
    """
    Custom callback for plotting additional values in tensorboard.
    """

    def __init__(self, verbose=0, info_keywords=()):
        super().__init__(verbose)
        self.returns_info = {
            "results/return": [],
            "results/episode_length": [],
            "results/success": [],
            "results/success_subtasks": [],
        }

    def _on_step(self) -> bool:
        infos = self.locals["infos"]
        for idx in range(len(infos)):
            curr_info = infos[idx]
            if "episode" in curr_info:
                self.returns_info["results/return"].append(curr_info["episode"]["r"])
                self.returns_info["results/episode_length"].append(curr_info["episode"]["l"])
                cur_info_success = 0
                if "success" in curr_info:
                    cur_info_success = curr_info["success"]
                self.returns_info["results/success"].append(cur_info_success)
                cur_info_success_subtasks = 0
                if "success_subtasks" in curr_info:
                    cur_info_success_subtasks = curr_info["success_subtasks"]
                self.returns_info["results/success_subtasks"].append(cur_info_success_subtasks)
        return True

    def _on_rollout_end(self) -> None:
        
        for key in self.returns_info.keys():
            if self.returns_info[key]:
                self.logger.record(key, np.mean(self.returns_info[key]))
                self.returns_info[key] = []


def main(argv):

    env = SubprocVecEnv([make_env(i) for i in range(ARGS.num_envs)])
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0)

    EVAL_SEED_OFFSET = 1000
    model_save_path = f"models/{ARGS.env_name}/{ARGS.exp_name}"
    os.makedirs(model_save_path, exist_ok=True)

    eval_env = DummyVecEnv([make_env(EVAL_SEED_OFFSET)])
    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False, clip_obs=10.0, training=False)
    best_model_callback = CustomEvalCallback(
        eval_env=eval_env,
        stage_thresholds=[float(x.strip()) for x in ARGS.stage_thresholds.split(",")],
        best_model_save_path=model_save_path,
        callback_on_new_best=BestModelHandler(
            eval_env=eval_env, 
            save_path=os.path.join(model_save_path, "best_vecnormalize.pkl"),
            render_steps=1000
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
        tags=[f"seed_{ARGS.seed}", f"exp_{ARGS.exp_name}"],
        sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
        monitor_gym=True,  # auto-upload the videos of agents playing the game
        save_code=False,  # optional
        config=vars(ARGS)
    )

    model = PPO(
        "MlpPolicy", 
        env, 
        verbose=1, 
        tensorboard_log=f"runs/baseline_{ARGS.env_name}_{ARGS.exp_name}", 
        learning_rate=float(ARGS.learning_rate), 
        n_steps=2048,
        batch_size=512,
        policy_kwargs=dict(
            net_arch=dict(
                pi=[256, 256],
                vf=[256, 256],
            )
        ),
        seed=ARGS.seed,
        device="cuda"
    )
    
    try:
        model.learn(
            total_timesteps=ARGS.max_steps, 
            log_interval=1, 
            callback=[
                best_model_callback,
                LogCallback(info_keywords=[]), 
                EpisodeLogCallback()
            ]
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

if __name__ == '__main__':
#   app.run(main)
    main(None)