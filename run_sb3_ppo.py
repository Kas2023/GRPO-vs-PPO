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

from utils.wrappers import DoorCurriculumWrapper, ActionSmoothnessWrapper, ControlCostWrapper

parser = argparse.ArgumentParser()
parser.add_argument("--env_name", type=str)
parser.add_argument("--exp_name", type=str)
parser.add_argument("--seed", type=int)
parser.add_argument("--num_envs", default=4, type=int)
parser.add_argument("--learning_rate", default=3e-5, type=float)
parser.add_argument("--max_steps", default=20000000, type=int)
parser.add_argument("--eval_freq", default=100000, type=int)
parser.add_argument("--n_eval_episodes", default=10, type=int)
parser.add_argument("--load_pretrained", action="store_true")
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
        # env = ControlCostWrapper(env, penalty_coef=0.02, force_margin=5.0)
        # env = ActionSmoothnessWrapper(env, penalty_coef=0.001)
        env = DoorCurriculumWrapper(env, stage=1)
        env = TimeLimit(env, max_episode_steps=1000)
        env = Monitor(env, f"logs/{ARGS.env_name}/{ARGS.exp_name}/train/env_{rank}")
        env.reset(seed=ARGS.seed + rank)
        
        return env

    return _init


def splice_pretrained_into_model(model, pt_path, mean_path, var_path):
    # 1. 加载 55 维的预训练权重
    pretrained_dict = th.load(pt_path)
    current_dict = model.policy.state_dict()
    
    print("Splicing weights...")
    for key in pretrained_dict.keys():
        if key in current_dict:
            if pretrained_dict[key].shape == current_dict[key].shape:
                # 隐藏层维度相同，直接拷贝
                current_dict[key] = pretrained_dict[key]
            else:
                # 第一层权重维度不匹配 [256, 155] vs [256, 55]
                print(f"Splicing layer: {key}")
                # 只拷贝前 55 维对应的列
                current_dict[key][:, :55] = pretrained_dict[key]
                # 后 100 维保持随机初始化的微小权重，或者设为 0
    
    model.policy.load_state_dict(current_dict)

    # 2. 同步归一化统计量 (VecNormalize)
    # 同样的逻辑：前 55 维用预训练的，后 100 维用默认的（均值0，方差1）
    pt_mean = np.load(mean_path)[-1, :]
    pt_var = np.load(var_path)[-1, :]
    
    train_env = model.get_vec_normalize_env()
    
    # 准备 155 维的均值和方差
    new_mean = np.zeros(155)
    new_var = np.ones(155)
    
    new_mean[:55] = pt_mean
    new_var[:55] = pt_var
    
    train_env.obs_rms.mean = new_mean
    train_env.obs_rms.var = new_var
    train_env.obs_rms.count = 1e3
    
    print("Bootstrap successful!")


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
        # video = []
        # obs = self.eval_env.reset()
        # for _ in range(self.render_steps):
        #     action, _ = self.model.predict(obs, deterministic=True)
        #     obs, _, _, _ = self.eval_env.step(action)
        #     pixels = self.eval_env.render()
        #     video.append(pixels.transpose(2, 0, 1)) # HWC -> CHW

        # video_array = np.stack(video)
        # wandb.log({
        #     "results/best_video": wandb.Video(video_array, fps=50, format="gif"),
        # })
        return True


class CustomEvalCallback(EvalCallback):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.current_stage = 1
        self.model_save_path = kwargs.get('best_model_save_path', None)

    def _on_step(self) -> bool:
        # 1. 执行原有的同步和评估逻辑
        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            # 同步 VecNormalize
            train_vec_norm = self.model.get_vec_normalize_env()
            # if train_vec_norm is not None and isinstance(self.eval_env, VecNormalize):
            if train_vec_norm is not None and hasattr(self.eval_env, 'obs_rms'):
                self.eval_env.obs_rms = train_vec_norm.obs_rms
                self.eval_env.ret_rms = train_vec_norm.ret_rms
            
            # 2. 调用父类评估 (会更新 self.last_mean_reward)
            continue_training = super()._on_step()

            # 3. 保存模型参数
            self.model.save(f"{self.model_save_path}/latest_model.zip")
            train_vec_norm.save(f"{self.model_save_path}/latest_vecnormalize.pkl")
            
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


class CurriculumLogCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_metrics = {
            "hand_distances": [],
            "hatch_angles": [],
            "door_openness": [],
            "distance_from_door": [],
            "hand_hooking": [],
            "robot_x": [],
        }
    
    def _on_step(self) -> bool:
        infos = self.locals["infos"]
        for idx, info in enumerate(infos):
            # 收集每个 step 的物理量
            if "hand_distance" in info:
                self.episode_metrics["hand_distances"].append(info["hand_distance"])
                self.episode_metrics["hatch_angles"].append(info["hatch_angle"])
                self.episode_metrics["door_openness"].append(info["door_openness"])
                self.episode_metrics["distance_from_door"].append(info["distance_from_door"])
                self.episode_metrics["hand_hooking"].append(info["hand_hooking"])
                self.episode_metrics["robot_x"].append(info["robot_x"])
            if self.locals['dones'][idx]:
                # episode 结束时，记录平均值
                wandb.log({
                    "curriculum/stage": info.get("curriculum_stage", 0),
                    "curriculum/avg_hand_distance": np.mean(self.episode_metrics["hand_distances"]),
                    "curriculum/avg_hatch_angle": np.mean(self.episode_metrics["hatch_angles"]),
                    "curriculum/avg_door_openness": np.mean(self.episode_metrics["door_openness"]),
                    "curriculum/max_robot_x": np.max(self.episode_metrics["robot_x"]),
                    "curriculum/distance_from_door": np.mean(self.episode_metrics["distance_from_door"]),
                    "curriculum/hand_hooking_rate": np.mean(self.episode_metrics["hand_hooking"]),
                })
                # 清空缓存
                self.episode_metrics = {k: [] for k in self.episode_metrics}
        
        return True


def main(argv):

    env = SubprocVecEnv([make_env(i) for i in range(ARGS.num_envs)])
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0)

    EVAL_SEED_OFFSET = 1000
    model_save_path = f"models/{ARGS.env_name}/{ARGS.exp_name}"
    os.makedirs(model_save_path, exist_ok=True)

    eval_env = DummyVecEnv([make_env(EVAL_SEED_OFFSET)])
    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False, clip_obs=10.0, training=False)

    # model = PPO(
    #     "MlpPolicy", 
    #     env, 
    #     verbose=1, 
    #     tensorboard_log=f"runs/baseline_{ARGS.env_name}_{ARGS.exp_name}", 
    #     learning_rate=float(ARGS.learning_rate), 
    #     n_steps=2048,
    #     batch_size=512,
    #     policy_kwargs=dict(
    #         net_arch=dict(
    #             pi=[256, 256],
    #             vf=[256, 256],
    #         )
    #     ),
    #     seed=ARGS.seed,
    #     device="cuda"
    # )

    # if ARGS.load_pretrained:
    #     splice_pretrained_into_model(
    #         model, 
    #         pt_path="humanoid-bench/data/reach_one_hand/torch_model.pt", 
    #         mean_path="humanoid-bench/data/reach_one_hand/mean.npy", 
    #         var_path="humanoid-bench/data/reach_one_hand/var.npy"
    #     )

    if ARGS.load_pretrained:
        model = PPO.load(
            "models/h1hand-door-v0/curriculum-v6.3/best_model.zip",
            tensorboard_log=f"runs/baseline_{ARGS.env_name}_{ARGS.exp_name}"
        )
        
        # env = VecNormalize.load("models/h1hand-door-v0/curriculum-v6.3/best_vecnormalize.pkl", env)
        env = VecNormalize.load("models/h1hand-door-v0/curriculum-v6.3/best_vecnormalize.pkl", env.venv)
        env.training = True
        
        eval_env.obs_rms = env.obs_rms
        eval_env.ret_rms = env.ret_rms
        eval_env.training  = False
        
        env.env_method("set_stage", 4) 
        eval_env.env_method("set_stage", 4)
        model.set_env(env)

    best_model_callback = CustomEvalCallback(
        eval_env=eval_env,
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

    try:
        model.learn(
            total_timesteps=ARGS.max_steps, 
            log_interval=1,
            reset_num_timesteps=False,
            callback=[
                best_model_callback,
                LogCallback(info_keywords=[]), 
                EpisodeLogCallback(),
                CurriculumLogCallback(),
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