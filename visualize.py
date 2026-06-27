# visualize.py 修改
import argparse
import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from gymnasium.wrappers import RecordVideo
import humanoid_bench

from utils.wrappers import DoorCurriculumWrapper

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_path", type=str, required=True)
    parser.add_argument("--env_name", type=str, default="h1hand-door-v0")
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--output_dir", type=str, default="./videos")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()

def main():
    args = parse_args()
    
    # 创建环境
    env = gym.make(args.env_name, render_mode="rgb_array")
    env = RecordVideo(env, args.output_dir, episode_trigger=lambda x: True, disable_logger=True)
    env = DoorCurriculumWrapper(env, stage=4)
    env.reset(seed=args.seed)

    # 包装成 DummyVecEnv（VecNormalize 需要）
    env = DummyVecEnv([lambda: env])

    # 加载模型
    model = PPO.load(f"{args.save_path}/best_model.zip", env=env)

    # 加载 VecNormalize 统计量
    env = VecNormalize.load(f"{args.save_path}/best_vecnormalize.pkl", env)
    env.training = False
    env.norm_reward = False

    # 注意：VecEnv 的 step 返回格式不同，需要调整
    for ep in range(args.episodes):
        obs = env.reset()
        done = False
        ep_return = 0
        ep_length = 0
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            ep_return += reward[0]  # VecEnv 返回的是数组
            ep_length += 1
            
            # 获取物理量（需要从 wrapper 获取，VecEnv 下需要特殊处理）
            if ep_length % 100 == 0 or done:
                # VecEnv 环境下获取 wrapper 的方法
                original_env = env.venv.envs[0].unwrapped
                if hasattr(original_env, 'get_physical_metrics'):
                    metrics = original_env.get_physical_metrics()
                    print(f"Step {ep_length:4d}: HandDist={metrics.get('hand_distance', 0):.3f}, "
                        f"HatchAng={metrics.get('hatch_angle', 0):.3f}, "
                        f"DoorOpen={metrics.get('door_openness', 0):.3f}, "
                        f"Return={ep_return:.2f}")
        
        print(f"Episode {ep+1}: Return={ep_return:.2f}, Length={ep_length}")
    
    env.close()
    print(f"\nVideos saved to: {args.output_dir}")

if __name__ == "__main__":
    main()