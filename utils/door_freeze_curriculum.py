"""
逆向课程学习 v2 — DoorFreezeCurriculumWrapper

核心思路（用户建议）:
- 不按正向课程（靠近→勾→拧→推→穿）绕远路
- 不按逆向课程 v1（初始门大开→门越来越闭）的方式，因为低 tau 阶段
  纯穿门会覆盖 v6.3 checkpoint 的开门技能
- v2 方案：监测 door_hinge 是否进入平稳期（连续 N 步不再增大），
  一旦检测到平稳则冻结门在当前 episode 达到的最大角度，
  给机器人充足时间从容穿过；随后逐步减少冻结时间
- 平稳检测 vs 固定阈值：机器人能推到什么程度就冻结在什么程度，
  不会因为达不到 0.7 而永远不触发冻结，训练更稳定

课程阶段:
  Stage 0: 检测到平稳后永久冻结（整个 episode 内不反弹）
  Stage 1: 检测到平稳后冻结 500 步，然后释放
  Stage 2: 冻结 250 步
  Stage 3: 冻结 100 步
  Stage 4: 不冻结（完整物理，门自然回弹）——等价于原始任务

每阶段推进条件: 滑窗 20 episodes 成功率 ≥ 70%
成功定义: robot_x > 1.0（穿过门）

与 InvertedDoorCurriculumWrapper 的关系:
- 两者互相替代，不同时使用
- v2 天然解决了 v1 的灾难性遗忘问题：每个 episode 都必须先开门
  才能触发冻结，因此开门技能在每个 episode 都被强制练习
"""

import numpy as np
import gymnasium as gym
import mujoco

# 各阶段的冻结步数（Stage 4 不冻结）
STAGE_FREEZE_DURATIONS = {
    0: 100000,  # 永久冻结（远超 max_episode_steps=1000）
    1: 500,
    2: 250,
    3: 100,
    4: 0,       # 不冻结，等价于原始任务
}

MAX_STAGE = 4


class DoorFreezeCurriculumWrapper(gym.Wrapper):
    """
    门冻结课程学习 Wrapper。

    机制：
    - 每个 episode 内，当 door_hinge (qpos[-2]) 首次达到 threshold (0.7)
      时，触发冻结。
    - 冻结期间：每步将 qpos[-2] clamp 回冻结角度，qvel[-2] 清零，
      mj_forward 同步状态。
    - 冻结持续 `STAGE_FREEZE_DURATIONS[stage]` 步后释放，
      门从冻结角度自然回弹（恢复完整物理）。
    - Stage 4 冻结时长为 0，等价于原始完整任务。
    """

    def __init__(
        self,
        env,
        stage=0,
        success_threshold=0.70,
        window_size=20,
        step_penalty=-0.01,
        truncation_penalty=-5.0,
        plateau_steps=20,
        min_freeze_angle=0.05,
    ):
        super().__init__(env)
        self.stage = stage
        self.success_threshold = success_threshold
        self.window_size = window_size
        self.step_penalty = step_penalty
        self.truncation_penalty = truncation_penalty
        self.plateau_steps = plateau_steps
        self.min_freeze_angle = min_freeze_angle

        # 滑窗成功追踪
        self._episode_successes = []

        # 当前 episode 内的冻结状态
        self._freeze_triggered = False       # 本 episode 是否已触发冻结
        self._frozen_hinge_angle = 0.0       # 触发冻结时的门角度（平稳期的最大值）
        self._freeze_remaining = 0           # 剩余冻结步数
        self._freeze_bonus_given = False     # 是否已发放冻结触发奖励

        # 平稳检测
        self._max_hinge_angle = 0.0          # 当前 episode 内门达到的最大角度
        self._steps_since_max = 0            # 自上次更新最大值以来经过的步数

        # 当前 episode 内的物理量统计
        self.episode_hand_distances = []
        self.episode_hatch_angles = []
        self.episode_door_openness = []
        self.episode_robot_x = []
        self.episode_distance_from_door = []
        self.episode_hand_hooking = []

        # 当前 episode 内是否已触发穿门奖励
        self._passed_door = False

        # 累计训练步数
        self._total_steps = 0

        # stage 变化历史
        self.stage_history = [(0, stage)]

    # ------------------------------------------------------------------
    # 冻结逻辑
    # ------------------------------------------------------------------

    def _get_freeze_duration(self):
        """返回当前 stage 的冻结步数"""
        return STAGE_FREEZE_DURATIONS.get(self.stage, 0)

    def _apply_freeze(self):
        """将门 clamp 回冻结角度，动能清零"""
        self.unwrapped.data.qpos[-2] = self._frozen_hinge_angle
        self.unwrapped.data.qvel[-2] = 0.0
        mujoco.mj_forward(self.unwrapped.model, self.unwrapped.data)

    # ------------------------------------------------------------------
    # 状态修改（reset 不修改初始 qpos，保持原始任务入口）
    # ------------------------------------------------------------------

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)

        # 清空 episode 追踪
        self._freeze_triggered = False
        self._frozen_hinge_angle = 0.0
        self._freeze_remaining = 0
        self._freeze_bonus_given = False
        self._passed_door = False
        self._max_hinge_angle = 0.0
        self._steps_since_max = 0

        self.episode_hand_distances = []
        self.episode_hatch_angles = []
        self.episode_door_openness = []
        self.episode_robot_x = []
        self.episode_distance_from_door = []
        self.episode_hand_hooking = []

        return obs, info

    # ------------------------------------------------------------------
    # 物理指标（自包含，不依赖 DoorCurriculumWrapper）
    # ------------------------------------------------------------------

    def get_physical_metrics(self):
        """获取当前环境的物理量"""
        task = self.env.unwrapped.task

        left_dist = np.linalg.norm(
            task._env.data.body("door_hatch").xpos
            - task._env.named.data.site_xpos["left_hand"]
        )
        right_dist = np.linalg.norm(
            task._env.data.body("door_hatch").xpos
            - task._env.named.data.site_xpos["right_hand"]
        )
        hand_dist = min(left_dist, right_dist)

        hatch_angle = task._env.data.qpos[-1]
        door_openness = task._env.data.qpos[-2]
        robot_x = task._env.named.data.site_xpos["imu", "x"]

        door_pos = task._env.data.body("door").xpos
        distance_from_door = np.linalg.norm(
            task._env.named.data.xpos["torso_link"] - door_pos
        )

        # 挂钩状态
        rod_center = (
            task._env.data.body("door_hatch").xpos
            + np.array([-0.141, -0.1, 0.0])
        )
        hand_pos = task.robot.right_hand_position()
        door_x = door_pos[0]

        hand_hooking = (
            rod_center[0] + 0.01 < hand_pos[0] < door_x - 0.05
            and (rod_center[1] - 0.05) < hand_pos[1] < rod_center[1]
            and abs(hand_pos[2] - rod_center[2]) < 0.08
        )

        return {
            "hand_distance": hand_dist,
            "hatch_angle": hatch_angle,
            "door_openness": door_openness,
            "robot_x": robot_x,
            "distance_from_door": distance_from_door,
            "hand_hooking": hand_hooking,
            "hand_pos": hand_pos,
            "door_x": door_x,
        }

    # ------------------------------------------------------------------
    # 奖励函数
    # ------------------------------------------------------------------

    def step(self, action):
        # 1. 底层环境步进
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._total_steps += 1

        # 2. 平稳检测 + 冻结触发
        hinge_angle = self.unwrapped.data.qpos[-2]
        freeze_duration = self._get_freeze_duration()

        # 追踪门角度的最大值
        if hinge_angle > self._max_hinge_angle:
            self._max_hinge_angle = hinge_angle
            self._steps_since_max = 0
        else:
            self._steps_since_max += 1

        # 平稳检测：连续 plateau_steps 步不再增大 → 触发冻结
        if (not self._freeze_triggered
                and freeze_duration > 0
                and self._steps_since_max >= self.plateau_steps
                and self._max_hinge_angle >= self.min_freeze_angle):
            self._freeze_triggered = True
            self._frozen_hinge_angle = self._max_hinge_angle
            self._freeze_remaining = freeze_duration

        # 3. 应用冻结（修改 qpos/qvel → mj_forward）
        if self._freeze_triggered and self._freeze_remaining > 0:
            self._apply_freeze()
            self._freeze_remaining -= 1
            # 重新获取观测以反映冻结后的门状态
            obs = self.unwrapped.task.get_obs()

        # 4. 收集物理指标
        metrics = self.get_physical_metrics()
        self.episode_hand_distances.append(metrics["hand_distance"])
        self.episode_hatch_angles.append(metrics["hatch_angle"])
        self.episode_door_openness.append(metrics["door_openness"])
        self.episode_robot_x.append(metrics["robot_x"])
        self.episode_distance_from_door.append(metrics["distance_from_door"])
        self.episode_hand_hooking.append(1 if metrics["hand_hooking"] else 0)

        # 5. 获取底层奖励分量
        task = self.env.unwrapped.task
        _, reward_dict = task.get_reward()

        s_r = reward_dict.get("stand_reward", 0.0)
        ctrl = reward_dict.get("small_control", 0.0)
        open_r = reward_dict.get("door_openness_reward", 0.0)
        hatch_r = reward_dict.get("door_hatch_openness_reward", 0.0)
        pass_r = reward_dict.get("passage_reward", 0.0)

        # ---- 奖励设计 ----

        # # 基础项（站立 + 控制代价 + 生存惩罚）
        # custom_reward = (
        #     0.1 * s_r
        #     - 0.05 * (1.0 - ctrl)
        #     + self.step_penalty
        # )

        # # 开门能力保持（始终存在，防止遗忘 v6.3 学会的钩门+拧闩+推门技能）
        # custom_reward += 0.3 * (5.0 * open_r + 0.1 * hatch_r)
        stand_bonus = 0.1 * s_r if s_r > 0.5 else 0
        hooking_bonus = 0.1 if metrics["hand_hooking"] else 0.0
        custom_reward = (
            stand_bonus 
            - 0.05 * (1 - ctrl) 
            + self.step_penalty 
        )

        # 穿门稠密引导
        custom_reward += 1.0 * pass_r

        door_openness = metrics["door_openness"]
        robot_x = metrics["robot_x"]
        door_x = metrics["door_x"]
        hatch_angle = metrics["hatch_angle"]

        # ---- 负向约束（v7.2 经验：少量约束有助于引导） ----

        # # 门没开时，禁止远离门（防止机器人不试图开门）
        # if door_openness < 0.2 and robot_x < door_x - 0.5:
        #     custom_reward -= 2.0

        # 门打开且闩解锁后，若明显后退则惩罚
        # if door_openness > 0.5 and hatch_angle > 0.75:
        #     if len(self.episode_robot_x) > 20:
        #         recent_x = self.episode_robot_x[-20:]
        #         if recent_x[-1] < recent_x[0] - 0.05:
        #             custom_reward -= 2.0

        if self._freeze_triggered:
        # if door_openness > 0.6:  # 门已开
            custom_reward += 0.2
            if len(self.episode_robot_x) > 20:
                # 检查最近20步的前进进度
                recent_x = self.episode_robot_x[-20:]
                progress = recent_x[-1] - recent_x[0]
                
                # 细分惩罚级别
                if progress < 0.002:  # 几乎完全不动
                    custom_reward -= 0.3
                elif progress < 0.005:  # 前进缓慢
                    custom_reward += 0.1
                elif progress < 0.01:  # 有点慢但不惩罚
                    custom_reward += 0.2
                else: # 正常前进
                    custom_reward += 0.5
            else:
                custom_reward -= 0.1
        else:
            custom_reward = (custom_reward
                + 2.0 * open_r
                + 0.1 * hatch_r 
                + hooking_bonus
            )

        # ---- 阶段性奖励 ----

        # 冻结触发奖励（一次性，鼓励开门到阈值）
        if self._freeze_triggered and not self._freeze_bonus_given and self.stage < 4:
            custom_reward += 5.0
            self._freeze_bonus_given = True

        # 穿门成功奖励（一次性，在整个 episode 内只发放一次）
        if robot_x > 1.0 and not self._passed_door:
            custom_reward += 50.0
            self._passed_door = True

        # 超时惩罚
        if truncated:
            custom_reward += self.truncation_penalty

        # ---- episode 结束时：检查成功并推进 stage ----
        if terminated or truncated:
            max_robot_x = max(self.episode_robot_x) if self.episode_robot_x else 0.0
            success = 1.0 if max_robot_x > 1.0 else 0.0
            self._episode_successes.append(success)
            if len(self._episode_successes) > self.window_size:
                self._episode_successes.pop(0)

            self._maybe_advance_stage()

            # 填入 info（兼容现有 logging callback）
            info["success"] = success
            info["success_subtasks"] = 1.0 if max(self.episode_door_openness) > 0.0 else 0.0
            info["curriculum_stage"] = self.stage
            info["freeze_stage"] = self.stage
            info["freeze_success_rate"] = (
                np.mean(self._episode_successes) if self._episode_successes else 0.0
            )
            info["freeze_triggered"] = float(self._freeze_triggered)
            info["freeze_max_hinge"] = self._max_hinge_angle

        # 物理量写入 info（供 CurriculumLogCallback 使用）
        info["hand_distance"] = metrics["hand_distance"]
        info["hatch_angle"] = metrics["hatch_angle"]
        info["door_openness"] = metrics["door_openness"]
        info["robot_x"] = metrics["robot_x"]
        info["distance_from_door"] = metrics["distance_from_door"]
        info["hand_hooking"] = metrics["hand_hooking"]
        info["hand_x"] = metrics["hand_pos"][0]

        return obs, custom_reward, terminated, truncated, info

    # ------------------------------------------------------------------
    # Stage 推进
    # ------------------------------------------------------------------

    def _maybe_advance_stage(self):
        """滑窗成功率达标后推进 stage"""
        if len(self._episode_successes) < self.window_size:
            return

        success_rate = np.mean(self._episode_successes)
        if success_rate >= self.success_threshold and self.stage < MAX_STAGE:
            old_stage = self.stage
            self.stage += 1
            self.stage_history.append((self._total_steps, self.stage))
            new_duration = STAGE_FREEZE_DURATIONS[self.stage]
            print(
                f"\n[FreezeCurriculum] Stage {old_stage} -> {self.stage} "
                f"(success_rate={success_rate:.2%}, "
                f"freeze_duration={new_duration}, "
                f"steps={self._total_steps})\n"
            )
            # 推进后重置滑窗，避免连续快速推进
            self._episode_successes = []

    def set_stage(self, stage):
        """外部手动设置 stage（供 eval 或恢复训练使用）"""
        self.stage = min(stage, MAX_STAGE)
        self._episode_successes = []
        duration = STAGE_FREEZE_DURATIONS[self.stage]
        print(
            f"[FreezeCurriculum] Stage set to {self.stage} (manual), "
            f"freeze_duration={duration}"
        )

    def get_stage_info(self):
        """返回 stage 相关统计信息"""
        return {
            "stage": self.stage,
            "freeze_duration": STAGE_FREEZE_DURATIONS[self.stage],
            "plateau_steps": self.plateau_steps,
            "min_freeze_angle": self.min_freeze_angle,
            "window_size": self.window_size,
            "current_successes": len(self._episode_successes),
            "current_success_rate": (
                np.mean(self._episode_successes) if self._episode_successes else 0.0
            ),
            "stage_history": self.stage_history,
        }
