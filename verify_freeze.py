"""验证 DoorFreezeCurriculumWrapper 的平稳检测冻结机制"""
import numpy as np
import gymnasium as gym
import humanoid_bench
from mujoco import mj_forward
from utils.door_freeze_curriculum import DoorFreezeCurriculumWrapper, STAGE_FREEZE_DURATIONS

print("=" * 60)
print("DoorFreezeCurriculumWrapper — 平稳检测验证")
print("=" * 60)


def run_until_freeze(env, action, max_steps=500):
    """循环步进直到冻结触发或超时，返回 (triggered, steps)"""
    for i in range(max_steps):
        obs, reward, terminated, truncated, info = env.step(action)
        if env._freeze_triggered:
            return True, i + 1
    return False, max_steps


# 测试1: 平稳检测触发冻结
print("\n--- 测试1: 平稳检测触发（手动设门到 0.75，等待平稳）---")
for stage in [0, 1, 2, 3]:
    duration = STAGE_FREEZE_DURATIONS[stage]
    env = gym.make("h1hand-door-v0")
    env = DoorFreezeCurriculumWrapper(env, stage=stage)
    obs, info = env.reset(seed=42)
    act_dim = env.action_space.shape[0]

    # 手动将门推到 0.75 并跑 mj_forward
    data = env.unwrapped.data
    data.qpos[-2] = 0.75
    data.qpos[-1] = 0.3
    mj_forward(env.unwrapped.model, data)

    # 跑直到冻结触发（门弹簧振荡消退后平稳检测命中）
    triggered, steps = run_until_freeze(env, np.zeros(act_dim), max_steps=500)

    assert triggered, f"Stage {stage}: 应在 {env.plateau_steps} 步左右触发，跑了 {steps} 步未触发!"
    assert abs(env._frozen_hinge_angle - env._max_hinge_angle) < 0.01, \
        f"冻结角度应等于最大角度"
    print(f"  Stage {stage}: 平稳触发 ✓  max_hinge={env._max_hinge_angle:.3f}  "
          f"frozen_at={env._frozen_hinge_angle:.3f}  steps={steps}")

    env.close()

# 测试2: Stage 4 不触发冻结
print("\n--- 测试2: Stage 4 不触发 ---")
env = gym.make("h1hand-door-v0")
env = DoorFreezeCurriculumWrapper(env, stage=4)
obs, info = env.reset(seed=42)
act_dim = env.action_space.shape[0]

data = env.unwrapped.data
data.qpos[-2] = 0.75
data.qpos[-1] = 0.3
mj_forward(env.unwrapped.model, data)

triggered, steps = run_until_freeze(env, np.zeros(act_dim), max_steps=100)
assert not triggered, "Stage 4: 冻结不应触发!"
print(f"  Stage 4: 未触发 ✓  ({steps} steps)")

env.close()

# 测试3: 低于 min_freeze_angle 不触发
print("\n--- 测试3: 门几乎没动时不触发 (min_freeze_angle=0.05) ---")
env = gym.make("h1hand-door-v0")
env = DoorFreezeCurriculumWrapper(env, stage=0, min_freeze_angle=0.05)
obs, info = env.reset(seed=42)
act_dim = env.action_space.shape[0]

# 门保持在 ~0.02（低于 0.05 阈值）
data = env.unwrapped.data
data.qpos[-2] = 0.02
mj_forward(env.unwrapped.model, data)

triggered, steps = run_until_freeze(env, np.zeros(act_dim), max_steps=200)
assert not triggered, f"门角度 0.02 < 0.05: 不应触发! (max={env._max_hinge_angle:.3f})"
print(f"  低于阈值不触发 ✓  max_hinge={env._max_hinge_angle:.3f}")

env.close()

# 测试4: 不同角度都能触发
print("\n--- 测试4: 不同门角度均能触发 ---")
for angle in [0.15, 0.35, 0.55, 0.90]:
    env = gym.make("h1hand-door-v0")
    env = DoorFreezeCurriculumWrapper(env, stage=0)
    obs, info = env.reset(seed=42)
    act_dim = env.action_space.shape[0]

    data = env.unwrapped.data
    data.qpos[-2] = angle
    data.qpos[-1] = 0.3
    mj_forward(env.unwrapped.model, data)

    triggered, steps = run_until_freeze(env, np.zeros(act_dim), max_steps=500)
    assert triggered, f"角度 {angle}: 应触发!"
    # 注意：冻结角度可能小于设定值，因为第一步弹簧就会拉回
    print(f"  set={angle:.2f} → frozen_at={env._frozen_hinge_angle:.3f}  "
          f"steps={steps}  ✓")

    env.close()

# 测试5: max 追踪正确（递增后平稳）
print("\n--- 测试5: 门逐步推开，max 追踪正确 ---")
env = gym.make("h1hand-door-v0")
env = DoorFreezeCurriculumWrapper(env, stage=0)
obs, info = env.reset(seed=42)
act_dim = env.action_space.shape[0]

# 逐步手动增加门角度（模拟机器人逐步推开门）
for val in [0.05, 0.12, 0.18, 0.25, 0.30]:
    data = env.unwrapped.data
    data.qpos[-2] = val
    mj_forward(env.unwrapped.model, data)
    obs, reward, terminated, truncated, info = env.step(np.zeros(act_dim))

# 继续跑直到冻结
triggered, steps = run_until_freeze(env, np.zeros(act_dim), max_steps=500)
assert triggered, "平稳后应触发!"
assert 0.20 <= env._max_hinge_angle <= 0.35, \
    f"max 应在 0.25~0.30 附近 实际 {env._max_hinge_angle:.3f}"
assert env._frozen_hinge_angle == env._max_hinge_angle
print(f"  递增→平稳→冻结 ✓  max={env._max_hinge_angle:.3f}")

env.close()

# 测试6: 冻结过期释放
print(f"\n--- 测试6: 冻结过期释放 (Stage 3) ---")
env = gym.make("h1hand-door-v0")
env = DoorFreezeCurriculumWrapper(env, stage=3)
obs, info = env.reset(seed=42)
act_dim = env.action_space.shape[0]

data = env.unwrapped.data
data.qpos[-2] = 0.80
data.qpos[-1] = 0.4
mj_forward(env.unwrapped.model, data)

triggered, steps = run_until_freeze(env, np.zeros(act_dim), max_steps=500)
assert triggered, "应先触发冻结"
print(f"  冻结触发: frozen_at={env._frozen_hinge_angle:.3f} remaining={env._freeze_remaining}")

# 跑完冻结步
frozen_angle = env._frozen_hinge_angle
for i in range(env._freeze_remaining):
    obs, reward, terminated, truncated, info = env.step(np.zeros(act_dim))
    # 验证冻结期间门不动
    assert abs(env.unwrapped.data.qpos[-2] - frozen_angle) < 0.02, \
        f"冻结期间门移动: {env.unwrapped.data.qpos[-2]:.4f} != {frozen_angle:.4f}"

print(f"  冻结保持 ✓  remaining 耗尽")

# 释放后门自然回弹
hinge_before = env.unwrapped.data.qpos[-2]
for i in range(20):
    obs, reward, terminated, truncated, info = env.step(np.zeros(act_dim))
hinge_after = env.unwrapped.data.qpos[-2]
print(f"  释放前={hinge_before:.4f}  20步后={hinge_after:.4f}")
print(f"  门自然回弹 ✓" if hinge_after < hinge_before else f"  门已稳定或闩卡住")

env.close()

# 测试7: stage info
print(f"\n--- 测试7: Stage 信息 ---")
env = gym.make("h1hand-door-v0")
env = DoorFreezeCurriculumWrapper(env, stage=2)
info = env.get_stage_info()
print(f"  stage={info['stage']}  freeze_duration={info['freeze_duration']}  "
      f"plateau_steps={info['plateau_steps']}  "
      f"min_freeze_angle={info['min_freeze_angle']}  "
      f"success_rate={info['current_success_rate']:.2f}")
env.close()

print(f"\n{'=' * 60}")
print("所有验证通过! ✓")
print(f"{'=' * 60}")
