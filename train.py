import os
import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import EvalCallback
from grpo import GRPO
from group_strategy import KMeansGroupingCPU
from wrappers import ActionSmoothnessWrapper

def train_one_seed(
    algo_cls,
    env_id,
    seed,
    total_timesteps,
    algo_kwargs,
    n_eval_episodes=10,
    eval_freq=int(1e4)
):
    # ---------- directories ----------
    exp_name = f"{algo_cls.__name__}_seed{seed}"
    log_train = f"logs/{env_id}/{exp_name}/train/"
    log_eval = f"logs/{env_id}/{exp_name}/eval/"
    model_dir = f"models/{env_id}/{exp_name}/"

    os.makedirs(log_train, exist_ok=True)
    os.makedirs(log_eval, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    # ---------- envs ----------
    env_train = gym.make(env_id)
    env_train.reset(seed=seed)
    # env_train =  ActionSmoothnessWrapper(env_train)
    env_train = Monitor(env_train, log_train)

    env_eval = gym.make(env_id)
    env_eval.reset(seed=seed + 1000)
    # env_eval =  ActionSmoothnessWrapper(env_eval)
    env_eval = Monitor(env_eval, log_eval)

    # ---------- callback ----------
    eval_callback = EvalCallback(
        eval_env=env_eval,
        best_model_save_path=model_dir,
        log_path=log_eval,
        n_eval_episodes=n_eval_episodes,
        eval_freq=eval_freq,
        deterministic=True,
        render=False,
    )

    # ---------- model ----------
    if algo_cls.__name__ == "GRPO":
        model = algo_cls(
            "MlpPolicy",
            env_train,
            grouping_strategy=KMeansGroupingCPU(n_clusters=5),
            seed=seed,
            device="cpu",
            verbose=1,
            **algo_kwargs,
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

    # ---------- pre-train eval ----------
    mean_r, std_r = evaluate_policy(
        model, env_eval, n_eval_episodes=n_eval_episodes
    )
    print(f"[{exp_name}] Before training: {mean_r:.1f} ± {std_r:.1f}")

    # ---------- training ----------
    model.learn(
        total_timesteps=total_timesteps,
        callback=eval_callback,
        tb_log_name=algo_cls.__name__,
    )

    # ---------- load best & final eval ----------
    best_model_path = os.path.join(model_dir, "best_model.zip")
    best_model = algo_cls.load(best_model_path, env=env_eval)

    mean_r, std_r = evaluate_policy(
        best_model, env_eval, n_eval_episodes=n_eval_episodes
    )
    print(f"[{exp_name}] Best model: {mean_r:.1f} ± {std_r:.1f}")

    env_train.close()
    env_eval.close()

if __name__ == "__main__":
    ENV_ID = "BipedalWalker-v3"
    TOTAL_TIMESTEPS = int(1e6)
    # SEEDS = [42, 2024, 777, 88, 13]
    # SEEDS = [42, 2024, 777]
    SEEDS = [42]

    algo_kwargs = dict(
        tensorboard_log=f"./tensorboard/{ENV_ID}/",
        n_steps=2048,
        batch_size=64,
        learning_rate=3e-4,
    )

    for seed in SEEDS:
        train_one_seed(
            algo_cls=PPO,
            env_id=ENV_ID,
            seed=seed,
            total_timesteps=TOTAL_TIMESTEPS,
            algo_kwargs=algo_kwargs,
            n_eval_episodes=10,
            eval_freq=int(1e4)
        )
