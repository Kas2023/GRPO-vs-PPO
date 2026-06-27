"""
Fine-tune a pre-trained model using GRPO or PPO
"""

import os
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import EvalCallback
from grpo import GRPO
from wrappers import HardBipedalWalkerWrapper
from group_strategy import KMeansGroupingCPU
import torch.nn as nn

def finetune_one_seed(
    algo_cls,
    env_id,
    seed,
    model_path,  # 新增：预训练模型路径
    total_timesteps,
    algo_kwargs,
    n_eval_episodes=10,
):
    # ---------- directories ----------
    exp_name = f"{algo_cls.__name__}_seed{seed}_finetune"
    log_train = f"logs/{env_id}/{exp_name}/train/"
    log_eval = f"logs/{env_id}/{exp_name}/eval/"
    model_dir = f"models/{env_id}/{exp_name}/"

    os.makedirs(log_train, exist_ok=True)
    os.makedirs(log_eval, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    # ---------- envs ----------
    env_train = gym.make(env_id)
    env_train.reset(seed=seed)
    env_train = HardBipedalWalkerWrapper(env_train)
    env_train = Monitor(env_train, log_train)

    env_eval = gym.make(env_id)
    env_eval.reset(seed=seed + 1000)
    env_eval = HardBipedalWalkerWrapper(env_eval)
    env_eval = Monitor(env_eval, log_eval)

    # ---------- callback ----------
    eval_callback = EvalCallback(
        eval_env=env_eval,
        best_model_save_path=model_dir,
        log_path=log_eval,
        n_eval_episodes=n_eval_episodes,
        deterministic=True,
        render=False,
    )

    # ---------- load pre-trained model ----------
    if algo_cls.__name__ == "GRPO":
        model = GRPO(
            "MlpPolicy",
            env_train,
            grouping_strategy=KMeansGroupingCPU(n_clusters=12),
            seed=seed,
            device="cpu",
            verbose=1,
            **algo_kwargs
        )
    else:
        model = algo_cls(
            "MlpPolicy",
            env_train,
            seed=seed,
            device="cpu",
            verbose=1,
            **algo_kwargs,
        )
    baseline = PPO.load(model_path)
    model.policy.load_state_dict(baseline.policy.state_dict())
    
    # Override learning rate for fine-tuning (lower)
    model.learning_rate = algo_kwargs.get("learning_rate", 1e-4)
    for param_group in model.policy.optimizer.param_groups:
        param_group["lr"] = model.learning_rate

    # ---------- pre-train eval ----------
    mean_r, std_r = evaluate_policy(
        model, env_eval, n_eval_episodes=n_eval_episodes
    )
    print(f"[{exp_name}] Before fine-tuning: {mean_r:.1f} ± {std_r:.1f}")

    # ---------- fine-tuning ----------
    model.learn(
        total_timesteps=total_timesteps,
        callback=eval_callback,
        tb_log_name=f"{algo_cls.__name__}",
    )

    # ---------- load best & final eval ----------
    best_model_path = os.path.join(model_dir, "best_model.zip")
    if os.path.exists(best_model_path):
        best_model = algo_cls.load(best_model_path, env=env_eval)
        mean_r, std_r = evaluate_policy(
            best_model, env_eval, n_eval_episodes=n_eval_episodes
        )
        print(f"[{exp_name}] Best model after fine-tuning: {mean_r:.1f} ± {std_r:.1f}")

    env_train.close()
    env_eval.close()


if __name__ == "__main__":
    ENV_ID = "BipedalWalker-v3"
    TOTAL_TIMESTEPS = int(2e6)
    SEEDS = [42]

    algo_kwargs = dict(
        tensorboard_log=f"./tensorboard/{ENV_ID}_finetune/",
        n_steps=2048,
        batch_size=64,
        learning_rate=3e-5,  # 微调用更小的学习率
    )

    for seed in SEEDS:
        finetune_one_seed(
            algo_cls=PPO,  # 或 PPO
            env_id=ENV_ID,
            seed=seed,
            model_path=f"models/{ENV_ID}/PPO_seed42/best_model.zip",  # 预训练模型路径
            total_timesteps=TOTAL_TIMESTEPS,
            algo_kwargs=algo_kwargs,
            n_eval_episodes=10,
        )