"""
python eval.py \
    --exp_name curriculum-v7 \
    --stage 4 \
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

from utils.wrappers import DoorCurriculumWrapper


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_name", type=str, default="h1hand-door-v0")
    parser.add_argument("--exp_name", type=str, required=True)
    parser.add_argument("--model_dir", type=str, default="models")
    parser.add_argument("--model_file", type=str, default="best_model.zip")
    parser.add_argument("--vecnormalize_file", type=str, default="best_vecnormalize.pkl")
    parser.add_argument("--stage", type=int, default=3)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 2024, 777, 88, 13])
    parser.add_argument("--n_eval_episodes", type=int, default=20)
    return parser.parse_args()


def make_eval_env(env_name: str, seed: int, stage: int):
    """创建单进程评估环境（参考 visualize.py 的包装方式）"""
    env = gym.make(env_name)
    env = DoorCurriculumWrapper(env, stage=stage)
    env.reset(seed=seed)
    return env


def eval_single_seed(
    env_name: str,
    seed: int,
    model_path: str,
    vecnormalize_path: str,
    stage: int = 4,
    n_eval_episodes: int = 20,
):
    """单种子评估（参考 visualize.py 的模型加载方式）"""
    # 创建环境
    env = make_eval_env(env_name, seed, stage)
    env = DummyVecEnv([lambda: env])

    # 加载模型（传入 env 让 PPO 知道 obs/act space）
    model = PPO.load(model_path, env=env)

    # 加载 VecNormalize 统计量（参考 visualize.py）
    if os.path.exists(vecnormalize_path):
        env = VecNormalize.load(vecnormalize_path, env)
        env.training = False
        env.norm_reward = False
    else:
        print(f"[WARN] VecNormalize file not found: {vecnormalize_path}")

    success_count = 0
    episode_returns = []
    episode_lengths = []
    episode_robot_x = []

    for ep in range(n_eval_episodes):
        env.venv.envs[0].reset(seed=seed + ep)
        obs = env.reset()
        done = False
        ep_return = 0
        ep_length = 0
        ep_success = False
        ep_robot_x = []

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, dones, info = env.step(action)
            ep_return += reward[0]
            ep_length += 1
            ep_robot_x.append(info[0]["robot_x"])

            # 检查 success（info 是 tuple，取第一个环境的 info）
            env_info = info[0]
            if env_info.get("success", 0) == 1:
                ep_success = True

            done = dones[0]

        if ep_success:
            success_count += 1
        episode_returns.append(ep_return)
        episode_lengths.append(ep_length)
        episode_robot_x.append(np.max(ep_robot_x))

    env.close()

    success_rate = success_count / n_eval_episodes

    return {
        "seed": seed,
        "success_rate": success_rate,
        "success_count": success_count,
        "mean_return": np.mean(episode_returns),
        "std_return": np.std(episode_returns),
        "mean_length": np.mean(episode_lengths),
        "std_length": np.std(episode_lengths),
        "episode_robot_x": episode_robot_x,
    }


def eval_multi_seed(
    env_name: str,
    exp_name: str,
    model_dir: str = "models",
    model_file: str = "best_model.zip",
    vecnormalize_file: str = "best_vecnormalize.pkl",
    stage: int = 3,
    seeds: list = None,
    n_eval_episodes: int = 20,
):
    """多种子评估（结构参考 eval_old.py）"""
    if seeds is None:
        seeds = [42, 2024, 777, 88, 13]

    model_path = os.path.join(model_dir, env_name, exp_name, model_file)
    vecnormalize_path = os.path.join(model_dir, env_name, exp_name, vecnormalize_file)

    print(f"Model      : {model_path}")
    print(f"VecNorm    : {vecnormalize_path}")
    print(f"Seeds      : {seeds}")
    print(f"Episodes   : {n_eval_episodes} per seed")
    print(f"Stage      : {stage}")
    print("-" * 50)

    results = []
    for seed in seeds:
        metrics = eval_single_seed(
            env_name=env_name,
            seed=seed,
            model_path=model_path,
            vecnormalize_path=vecnormalize_path,
            stage=stage,
            n_eval_episodes=n_eval_episodes,
        )
        results.append(metrics)

        print(f"[seed {seed:4d}] "
              f"success_rate={metrics['success_rate']:.3f} "
              f"({metrics['success_count']}/{n_eval_episodes}) | "
              f"return={metrics['mean_return']:.1f} +/- {metrics['std_return']:.1f} | "
              f"length={metrics['mean_length']:.0f} +/- {metrics['std_length']:.0f}")
        print(f"{metrics['episode_robot_x']}")

    # 汇总
    success_rates = [r["success_rate"] for r in results]
    returns = [r["mean_return"] for r in results]

    print("-" * 50)
    print(f"Aggregate over {len(seeds)} seeds:")
    print(f"  Success rate : {np.mean(success_rates):.3f} +/- {np.std(success_rates):.3f}")
    print(f"  Mean return  : {np.mean(returns):.1f} +/- {np.std(returns):.1f}")

    return results


if __name__ == "__main__":
    args = parse_args()
    eval_multi_seed(
        env_name=args.env_name,
        exp_name=args.exp_name,
        model_dir=args.model_dir,
        model_file=args.model_file,
        vecnormalize_file=args.vecnormalize_file,
        stage=args.stage,
        seeds=args.seeds,
        n_eval_episodes=args.n_eval_episodes,
    )
