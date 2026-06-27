import gymnasium as gym
import humanoid_bench
import numpy as np
from gymnasium.wrappers import RecordVideo

env = gym.make("h1hand-door-v0", render_mode="rgb_array")
env = RecordVideo(env, "./door_simulation_videos", episode_trigger=lambda x: True, disable_logger=True)
env.reset()

door_joint_idx = -2
hatch_joint_idx = -1

print("=" * 50)
print("模拟：插销按下 + 拉门")
print("=" * 50)

while True:
    door_angle = env.unwrapped.task._env.data.qpos[door_joint_idx]
    hatch_angle = env.unwrapped.task._env.data.qpos[hatch_joint_idx]
    
    print(f"\n当前: 门={door_angle:.3f} rad, 插销={hatch_angle:.3f} rad")
    
    cmd = input("\n[1]插销设1.0 [2]插销设1.5 [3]门拉力+100 [4]门拉力-100 [5]重置 [q]退出: ")
    
    if cmd == 'q':
        break
    elif cmd == '1':
        env.unwrapped.task._env.data.qpos[hatch_joint_idx] = 1.0
        env.unwrapped.task._env.data.qfrc_applied[hatch_joint_idx] = 50.0
        print("插销角度设为 1.0 rad")
    elif cmd == '2':
        env.unwrapped.task._env.data.qpos[hatch_joint_idx] = 1.5
        print("插销角度设为 1.5 rad")
    elif cmd == '3':
        env.unwrapped.task._env.data.qfrc_applied[door_joint_idx] = 100.0
        env.unwrapped.task._env.data.qfrc_applied[hatch_joint_idx] = -10.0
        print("给门施加 +100 力矩（拉）")
    elif cmd == '4':
        env.unwrapped.task._env.data.qfrc_applied[door_joint_idx] = -100.0
        print("给门施加 -100 力矩（推）")
    elif cmd == '5':
        env.unwrapped.task._env.data.qpos[door_joint_idx] = 0.0
        env.unwrapped.task._env.data.qvel[door_joint_idx] = 0.0
        env.unwrapped.task._env.data.qpos[hatch_joint_idx] = 0.0
        print("重置门和插销")
    
    # 模拟 30 步
    for step in range(10000):
        env.step(np.zeros(env.action_space.shape[0]))
        if step % 10 == 0:
            new_door = env.unwrapped.task._env.data.qpos[door_joint_idx]
            new_hatch = env.unwrapped.task._env.data.qpos[hatch_joint_idx]
            print(f"  step {step}: 门={new_door:.3f}, 插销={new_hatch:.3f}")
    
    env.unwrapped.task._env.data.qfrc_applied[:] = 0.0

env.close()