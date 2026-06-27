import os
import numpy as np
import pandas as pd
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from grpo import GRPO

def eval_single_seed(
    algo_cls,
    env_id: str,
    seed: int,
    model_path: str,
    monitor_csv: str,
    n_eval_episodes: int = 20,
    reward_threshold: float = 200,
    smooth_window: int = 100,
):
    env = gym.make(env_id)
    env.reset(seed=seed + 5000)
    model = algo_cls.load(model_path, env=env)

    # 1. Final policy performance
    mean_r, std_r = evaluate_policy(
        model,
        env,
        n_eval_episodes=n_eval_episodes,
        deterministic=True,
        render=False
    )

    # 2. Sample efficiency from Monitor
    df = pd.read_csv(monitor_csv, skiprows=1)

    rewards = df["r"].to_numpy()
    lengths = df["l"].to_numpy()
    times = df["t"].to_numpy()

    cum_steps = np.cumsum(lengths)

    steps_to_threshold = None
    time_to_threshold = None

    if len(rewards) >= smooth_window:
        smoothed = np.convolve(
            rewards,
            np.ones(smooth_window) / smooth_window,
            mode="valid"
        )
        idx = np.where(smoothed >= reward_threshold)[0]
        if len(idx) > 0:
            ep_idx = idx[0] + smooth_window - 1
            steps_to_threshold = int(cum_steps[ep_idx])
            time_to_threshold = float(times[ep_idx])

    env.close()

    return {
        "mean_reward": mean_r,
        "std_reward": std_r,
        "steps_to_threshold": steps_to_threshold,
        "time_to_threshold": time_to_threshold
    }


def eval_multi_seed(
    algo_cls,
    env_id: str,
    seeds: list,
    base_model_dir: str = "models/",
    base_log_dir: str = "logs/",
):
    results = []
    algo_name = algo_cls.__name__

    for seed in seeds:
        model_path = os.path.join(
            base_model_dir, f"{algo_name}_seed{seed}", "best_model.zip"
        )
        monitor_csv = os.path.join(
            base_log_dir, f"{algo_name}_seed{seed}", "train", "monitor.csv"
        )

        metrics = eval_single_seed(
            algo_cls=algo_cls,
            env_id=env_id,
            seed=seed,
            model_path=model_path,
            monitor_csv=monitor_csv
        )
        # metrics["seed"] = seed
        results.append(metrics)

        print(f"\n[{algo_name} | seed {seed}]")
        print(f"Mean reward        : {metrics['mean_reward']:.2f}")
        print(f"Std reward         : {metrics['std_reward']:.2f}")
        print(f"Steps to threshold : {metrics['steps_to_threshold']}")
        print(f"Time to threshold  : {metrics['time_to_threshold']} s")

    df = pd.DataFrame(results)

    print("\n===== Aggregate over seeds =====")
    print(df.mean(numeric_only=True))
    print("\n===== Std over seeds =====")
    print(df.std(numeric_only=True))

    return df


if __name__ == "__main__":
    ENV_ID = "InvertedPendulum-v4"
    # SEEDS = [42, 2024, 777, 88, 13]
    SEEDS = [42, 2024, 777]

    eval_multi_seed(
        algo_cls=GRPO,
        env_id=ENV_ID,
        seeds=SEEDS
    )
