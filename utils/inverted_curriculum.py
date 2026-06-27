"""
逆向课程学习 Wrapper — InvertedDoorCurriculumWrapper

核心思路（导师建议）:
- 不按正向课程（靠近→勾→拧→推→穿）训练
- 反过来：门一开始大开着，机器人站在门口附近，先学会"穿门"
- 然后逐步关小门、逐步把机器人放远
- 单参数 tau (0→1) 连续控制初始难度

与 DoorCurriculumWrapper 的关系:
- 两者互相替代，不同时使用
- 逆向课程保留打开门相关的奖励信号，防止从 pretrained checkpoint
  加载的开门技能被遗忘
"""

import numpy as np
import gymnasium as gym
import mujoco


class InvertedDoorCurriculumWrapper(gym.Wrapper):
    """
    逆向课程学习 Wrapper。

    在 reset 时根据 tau 修改初始 qpos：
      - qpos[-2] (door_hinge):      easy=1.2, hard=0.0
      - qpos[-1] (door_hatch_hinge): easy=0.8, hard=0.0
      - qpos[0]  (robot pelvis x):   easy=0.5, hard=0.0

    tau 从 0 到 1 线性插值，tau=0 最简单，tau=1.0 等价于原始完整任务。
    """

    def __init__(
        self,
        env,
        initial_tau=0.0,
        tau_increment=0.05,
        success_threshold=0.40,
        window_size=20,
        step_penalty=-0.01,
        truncation_penalty=-5.0,
        # tau=0 时的初始值（最简单）
        max_door_hinge=1.2,
        max_door_hatch=0.8,
        near_door_robot_x=0.5,
    ):
        super().__init__(env)
        self.tau = initial_tau
        self.tau_increment = tau_increment
        self.success_threshold = success_threshold
        self.window_size = window_size
        self.step_penalty = step_penalty
        self.truncation_penalty = truncation_penalty

        # tau=0 时的 easy 值
        self.max_door_hinge = max_door_hinge
        self.max_door_hatch = max_door_hatch
        self.near_door_robot_x = near_door_robot_x

        # 记录 tau 的变化历史（供外部查询）
        self.tau_history = [(0, initial_tau)]  # (step_count, tau)

        # 滑窗成功追踪
        self._episode_successes = []  # 最近 N 个 episode 的成功标记

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

    # ------------------------------------------------------------------
    # 状态修改
    # ------------------------------------------------------------------

    def _interpolate_qpos(self):
        """根据当前 tau 计算三个需要修改的 qpos 值"""
        t = self.tau
        door_hinge = self.max_door_hinge * (1 - t)   # 1.2 → 0
        door_hatch = self.max_door_hatch * (1 - t)    # 0.8 → 0
        robot_x = self.near_door_robot_x * (1 - t)     # 0.5 → 0
        return door_hinge, door_hatch, robot_x

    def _get_modified_qpos(self):
        """返回修改后的完整 qpos 数组"""
        qpos = self.unwrapped.data.qpos.copy()
        door_hinge, door_hatch, robot_x = self._interpolate_qpos()
        qpos[-2] = door_hinge   # door_hinge
        qpos[-1] = door_hatch   # door_hatch_hinge
        qpos[0] = robot_x       # robot pelvis x
        return qpos

    def reset(self, **kwargs):
        # 1. 标准 reset（到 keyframe + 随机噪声）
        obs, info = self.env.reset(**kwargs)

        # 2. 修改 qpos 为逆向课程初始状态
        new_qpos = self._get_modified_qpos()
        self.unwrapped.set_state(new_qpos, self.unwrapped.data.qvel)
        mujoco.mj_forward(self.unwrapped.model, self.unwrapped.data)

        # 3. 重新获取观测（qpos 已变）
        obs = self.unwrapped.task.get_obs()

        # 4. 清空 episode 追踪
        self.episode_hand_distances = []
        self.episode_hatch_angles = []
        self.episode_door_openness = []
        self.episode_robot_x = []
        self.episode_distance_from_door = []
        self.episode_hand_hooking = []
        self._passed_door = False

        return obs, info

    # ------------------------------------------------------------------
    # 物理指标（从 DoorCurriculumWrapper 复制，自包含）
    # ------------------------------------------------------------------

    def get_physical_metrics(self):
        """获取当前环境的物理量"""
        task = self.env.unwrapped.task
        # 手到插销的距离（取左右手最小值）
        left_dist = np.linalg.norm(
            task._env.data.body("door_hatch").xpos
            - task._env.named.data.site_xpos["left_hand"]
        )
        right_dist = np.linalg.norm(
            task._env.data.body("door_hatch").xpos
            - task._env.named.data.site_xpos["right_hand"]
        )
        hand_dist = min(left_dist, right_dist)

        # 插销打开角度 (qpos[-1])
        hatch_angle = task._env.data.qpos[-1]

        # 门打开程度 (qpos[-2])
        door_openness = task._env.data.qpos[-2]

        # 机器人x位置 (通过门)
        robot_x = task._env.named.data.site_xpos["imu", "x"]

        # 离门距离
        door_pos = task._env.data.body("door").xpos
        distance_from_door = np.linalg.norm(
            task._env.named.data.xpos["torso_link"]
            - door_pos
        )

        # 挂钩状态
        rod_center = (
            task._env.data.body("door_hatch").xpos
            + np.array([-0.141, -0.1, 0.0])
        )
        hand_pos = task.robot.right_hand_position()

        hand_hooking = (
            rod_center[0] + 0.01 < hand_pos[0] < door_pos[0] - 0.05
            and (rod_center[1] - 0.05) < hand_pos[1] < rod_center[1]
            and abs(hand_pos[2] - rod_center[2]) < 0.08
        )

        door_x = door_pos[0]

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
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._total_steps += 1

        # 收集物理指标
        metrics = self.get_physical_metrics()
        self.episode_hand_distances.append(metrics["hand_distance"])
        self.episode_hatch_angles.append(metrics["hatch_angle"])
        self.episode_door_openness.append(metrics["door_openness"])
        self.episode_robot_x.append(metrics["robot_x"])
        self.episode_distance_from_door.append(metrics["distance_from_door"])
        self.episode_hand_hooking.append(1 if metrics["hand_hooking"] else 0)

        # 获取底层奖励分量
        task = self.env.unwrapped.task
        _, reward_dict = task.get_reward()

        s_r = reward_dict["stand_reward"]
        ctrl = reward_dict["small_control"]
        open_r = reward_dict["door_openness_reward"]
        hatch_r = reward_dict["door_hatch_openness_reward"]
        pass_r = reward_dict["passage_reward"]

        # ---- 奖励设计（融合 v7.2 penalty-dominant + 开门能力保持）----

        # 基础项
        custom_reward = (
            0.1 * s_r
            - 0.05 * (1 - ctrl)
            + self.step_penalty
        )

        # 开门能力保持（始终存在，防止遗忘 v6.3 学会的技能）
        custom_reward += 0.3 * (5.0 * open_r + 0.1 * hatch_r)

        # 穿门稠密奖励
        custom_reward += 1.0 * pass_r

        door_openness = metrics["door_openness"]
        hatch_angle = metrics["hatch_angle"]
        robot_x = metrics["robot_x"]
        door_x = metrics["door_x"]

        # ---- v7.2 风格负向约束 ----

        # 约束 1: 门没开时，禁止远离门
        if door_openness < 0.2 and robot_x < door_x - 0.5:
            custom_reward -= 2.0

        # 约束 2: 门打开且门闩解锁后，禁止不前进或后退
        if door_openness > 0.5 and hatch_angle > 0.75:
            if len(self.episode_robot_x) > 20:
                recent_x = self.episode_robot_x[-20:]
                progress = max(recent_x) - min(recent_x)

                if progress < 0.02:   # 几乎完全不动
                    custom_reward -= 3.0
                elif progress < 0.05:  # 前进缓慢
                    custom_reward -= 1.0
                # progress >= 0.05: 不惩罚

                # 后退惩罚
                if recent_x[-1] < recent_x[0] - 0.05:
                    custom_reward -= 2.0

        # 一次性穿门奖励
        if robot_x > 1.0 and not self._passed_door:
            custom_reward += 50.0
            self._passed_door = True

        # 超时惩罚
        if truncated:
            custom_reward += self.truncation_penalty

        # ---- episode 结束时：检查成功并推进 tau ----
        if terminated or truncated:
            max_robot_x = max(self.episode_robot_x) if self.episode_robot_x else 0.0
            success = 1.0 if max_robot_x > 1.0 else 0.0
            self._episode_successes.append(success)
            if len(self._episode_successes) > self.window_size:
                self._episode_successes.pop(0)

            self._maybe_advance_tau()

            # 填入 info（兼容现有 logging callback）
            info["success"] = success
            info["success_subtasks"] = 1.0 if max(self.episode_door_openness) > 0.0 else 0.0
            info["curriculum_stage"] = int(self.tau * 5)  # 把 tau 映射为 stage 用于日志
            info["inverted_tau"] = self.tau
            info["inverted_success_rate"] = (
                np.mean(self._episode_successes) if self._episode_successes else 0.0
            )

        # 物理量写入 info
        info["hand_distance"] = metrics["hand_distance"]
        info["hatch_angle"] = metrics["hatch_angle"]
        info["door_openness"] = metrics["door_openness"]
        info["robot_x"] = metrics["robot_x"]
        info["distance_from_door"] = metrics["distance_from_door"]
        info["hand_hooking"] = metrics["hand_hooking"]
        info["hand_x"] = metrics["hand_pos"][0]

        return obs, custom_reward, terminated, truncated, info

    # ------------------------------------------------------------------
    # Tau 推进
    # ------------------------------------------------------------------

    def _maybe_advance_tau(self):
        """滑窗成功率达标后推进 tau"""
        if len(self._episode_successes) < self.window_size:
            return

        success_rate = np.mean(self._episode_successes)
        if success_rate >= self.success_threshold and self.tau < 1.0:
            old_tau = self.tau
            self.tau = min(1.0, self.tau + self.tau_increment)
            self.tau_history.append((self._total_steps, self.tau))
            print(
                f"\n[InvertedCurriculum] tau {old_tau:.2f} -> {self.tau:.2f} "
                f"(success_rate={success_rate:.2%}, steps={self._total_steps})\n"
            )
            # 推进后重置滑窗，避免连续快速推进
            self._episode_successes = []

    def set_tau(self, tau):
        """外部手动设置 tau（供 eval 或恢复训练使用）"""
        self.tau = tau
        self._episode_successes = []
        print(f"[InvertedCurriculum] tau set to {self.tau:.2f} (manual)")

    def get_tau_info(self):
        """返回 tau 相关统计信息"""
        return {
            "tau": self.tau,
            "window_size": self.window_size,
            "current_successes": len(self._episode_successes),
            "current_success_rate": (
                np.mean(self._episode_successes) if self._episode_successes else 0.0
            ),
            "tau_history": self.tau_history,
        }
