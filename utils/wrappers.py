
import numpy as np
import gymnasium as gym

import numpy as np
import gymnasium as gym


class DoorCurriculumWrapper(gym.Wrapper):
    def __init__(
        self,
        env,
        stage=1,
        stage_thresholds=None,
        stage_success_episodes=5,
        step_penalty=-0.01,
        truncation_penalty=-5.0,
    ):
        super().__init__(env)
        self.stage = stage
        # 物理量阈值: [手-插销距离, 插销打开角度, 门打开程度]
        self.stage_thresholds = stage_thresholds or [0.3, 0.75, 0.4]
        self.stage_success_episodes = stage_success_episodes
        self.step_penalty = step_penalty
        self.truncation_penalty = truncation_penalty
        self.hooking_steps = 0  # 连续钩挂步数
        self.required_hooking_steps = 500  # 需要连续保持多少步（根据你的步长调整，约 0.5-1 秒）

        # 统计当前 episode 的物理量
        self.episode_hand_distances = []
        self.episode_hatch_angles = []
        self.episode_door_openness = []
        self.episode_robot_x = []
        self.episode_distance_from_door = []
        self.episode_hand_hooking = []

        # 阶段切换标记
        self.stage_just_changed = False
        self.consecutive_success = 0

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

    def check_stage_transition(self):
        """根据物理量判断是否切换阶段（需要连续 N 个 episode 达标）"""
        if self.stage == 1:
            hooking_ratio = np.mean(self.episode_hand_hooking)
            if hooking_ratio >= 0.6:  # 60% 时间在钩挂区域
                self.consecutive_success += 1
                print(f"Stage1: {self.consecutive_success}/{self.stage_success_episodes} consecutive episodes (hooking_ratio={hooking_ratio:.2f})")
                if self.consecutive_success >= self.stage_success_episodes:
                    self.stage = 2
                    print(f"\n[Curriculum] Stage 1 -> 2! Hooking ratio: {hooking_ratio:.2f}")
                    self.consecutive_success = 0
                    self.stage_just_changed = True
                    return True
            else:
                self.consecutive_success = 0

        elif self.stage == 2:
            avg_hatch_angle = np.mean(self.episode_hatch_angles) if self.episode_hatch_angles else 0.0
            if avg_hatch_angle >= self.stage_thresholds[1]:
                self.consecutive_success += 1
                print(f"Stage2: {self.consecutive_success}/{self.stage_success_episodes} consecutive successes")
                if self.consecutive_success >= self.stage_success_episodes:
                    self.stage = 3
                    print(f"\n[Curriculum] Stage 2 -> 3! Avg hatch angle: {avg_hatch_angle:.3f}")
                    self.consecutive_success = 0
                    self.stage_just_changed = True
                    return True
            else:
                self.consecutive_success = 0
        
        elif self.stage == 3:
            avg_door = np.mean(self.episode_door_openness) if self.episode_door_openness else 0.0
            if avg_door >= self.stage_thresholds[2]:
                self.consecutive_success += 1
                print(f"Stage3: {self.consecutive_success}/{self.stage_success_episodes} consecutive successes")
                if self.consecutive_success >= self.stage_success_episodes:
                    self.stage = 4
                    print(f"\n[Curriculum] Stage 3 -> 4! Avg door openness: {avg_door:.3f}")
                    self.consecutive_success = 0
                    self.stage_just_changed = True
                    return True
            else:
                self.consecutive_success = 0

        elif self.stage == 4:
            max_robot_x = np.max(self.episode_robot_x) if self.episode_robot_x else 0.0
            if max_robot_x >= 1.0:  # 通过门
                print(f"\n[Curriculum] Task completed! Robot passed door: {max_robot_x:.3f}")
                return True

        return False

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        # 重置 episode 统计
        self.episode_hand_distances = []
        self.episode_hatch_angles = []
        self.episode_door_openness = []
        self.episode_robot_x = []
        self.episode_distance_from_door = []
        self.episode_hand_hooking = []
        self.stage_just_changed = False
        if hasattr(self, 'passed_door'):
            del self.passed_door

        return obs

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        # 获取物理量
        metrics = self.get_physical_metrics()
        self.episode_hand_distances.append(metrics["hand_distance"])
        self.episode_hatch_angles.append(metrics["hatch_angle"])
        self.episode_door_openness.append(metrics["door_openness"])
        self.episode_robot_x.append(metrics["robot_x"])
        self.episode_distance_from_door.append(metrics["distance_from_door"])
        self.episode_hand_hooking.append(1 if metrics["hand_hooking"] else 0)

        # 获取各项奖励分量
        task = self.env.unwrapped.task
        _, reward_dict = task.get_reward()

        s_r = reward_dict["stand_reward"]
        ctrl = reward_dict["small_control"]
        open_r = reward_dict["door_openness_reward"]
        hatch_r = reward_dict["door_hatch_openness_reward"]
        prox_r = reward_dict["hand_hatch_proximity_reward"]
        pass_r = reward_dict["passage_reward"]

        # 基础时间惩罚
        step_penalty = self.step_penalty

        # 离门距离惩罚
        body_door_velocity = metrics["distance_from_door"] - self.episode_distance_from_door[-2] if len(self.episode_distance_from_door) > 1 else 0.0
        body_door_penalty = -0.2 * max(0, -body_door_velocity) if body_door_velocity < -0.05 else 0

        # ---------- 阶段课程学习 ----------
        if self.stage == 1:
            # 阶段1：专注靠近插销，站稳作为辅助，hooking一点点
            stand_bonus = 0.5 * s_r if s_r > 0.3 else 0
            hooking_bonus = 0.5 if metrics["hand_hooking"] else 0.0
            custom_reward = (
                stand_bonus 
                + 0.1 * prox_r 
                + hooking_bonus
                - 0.2 * (1 - ctrl) 
                + step_penalty 
                + body_door_penalty
            )

        elif self.stage == 2:
            stand_bonus = 0.3 * s_r if s_r > 0.5 else 0
            hooking_bonus = 0.2 if metrics["hand_hooking"] else 0.0
            custom_reward = (
                stand_bonus 
                + 0.5 * hatch_r 
                + hooking_bonus
                - 0.05 * (1 - ctrl) 
                + step_penalty 
            )

        elif self.stage == 3:
            stand_bonus = 0.1 * s_r if s_r > 0.5 else 0
            hooking_bonus = 0.1 if metrics["hand_hooking"] else 0.0
            custom_reward = (
                stand_bonus 
                + 5.0 * open_r
                + 0.1 * hatch_r 
                + hooking_bonus
                - 0.05 * (1 - ctrl) 
                + step_penalty 
            )
            # if metrics["door_openness"] < 0.3 and metrics["hatch_angle"] > 0.75:
                # pull_reward = max(0, metrics["door_openness"] - self.episode_door_openness[-2]) / 0.002 if len(self.episode_door_openness) > 1 else 0.0
                # custom_reward += pull_reward
            if metrics["hatch_angle"] > 0.75:
                sustain_reward = 50.0 * metrics["door_openness"] 
                custom_reward += sustain_reward

        elif self.stage == 4:
            # 基础项不变，适当减小open_r的影响
            stand_bonus = 0.1 * s_r if s_r > 0.5 else 0
            hooking_bonus = 0.1 if metrics["hand_hooking"] else 0.0
            custom_reward = (
                stand_bonus 
                + 2.0 * open_r
                + 0.1 * hatch_r 
                + hooking_bonus
                - 0.05 * (1 - ctrl) 
                + step_penalty 
            )
            
            door_openness = metrics["door_openness"]
            robot_x = metrics["robot_x"]
            door_x = metrics["door_x"]
            
            if door_openness > 0.6:  # 门已开
                custom_reward += 0.5
                if len(self.episode_robot_x) > 20:
                    # 检查最近20步的前进进度
                    recent_x = self.episode_robot_x[-20:]
                    progress = recent_x[-1] - recent_x[0]
                    
                    # 细分惩罚级别
                    if progress < 0.02:  # 几乎完全不动
                        custom_reward -= 3.0
                    elif progress < 0.05:  # 前进缓慢
                        custom_reward += 1.0
                    elif progress < 0.1:  # 有点慢但不惩罚
                        custom_reward += 2.0
                    else: # 正常前进
                        custom_reward += 5.0
            else:
                custom_reward -= 0.1
                    
            # 唯一的大奖励：成功穿过
            if robot_x > 1.0 and not hasattr(self, 'passed_door'):
                custom_reward += 50.0  # 一次性高奖励
                self.passed_door = True
            
            custom_reward += 1.0 * pass_r
        else:
            custom_reward = reward

        # 超时惩罚（防止拖延）
        if truncated:
            custom_reward += self.truncation_penalty

        # 检查阶段切换（在 episode 结束时）
        if terminated or truncated:
            self.check_stage_transition()
            info["curriculum_stage"] = self.stage
            info["stage_just_changed"] = self.stage_just_changed
            info["success"] = 1.0 if np.max(self.episode_robot_x) > 1.0 else 0.0
            info["success_subtasks"] = 1.0 if np.max(self.episode_door_openness) else 0.0

        # 添加物理量到 info 供 logging
        info["hand_distance"] = metrics["hand_distance"]
        info["hatch_angle"] = metrics["hatch_angle"]
        info["door_openness"] = metrics["door_openness"]
        info["robot_x"] = metrics["robot_x"]
        info["distance_from_door"] = metrics["distance_from_door"]
        info["hand_hooking"] = metrics["hand_hooking"]
        info["hand_x"] = metrics["hand_pos"][0]

        return obs, custom_reward, terminated, truncated, info
    
    def set_stage(self, stage):
        self.stage = stage
        print(f"Environment updated to Stage {self.stage}")
    
class ActionSmoothnessWrapper(gym.Wrapper):
    def __init__(self, env, penalty_coef=0.001):
        super().__init__(env)
        self.penalty_coef = penalty_coef
        self.prev_action = None

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        if self.prev_action is not None:
            # Action Rate Penalty: ||a_t - a_{t-1}||^2
            penalty = self.penalty_coef * np.sum(np.square(action - self.prev_action))
            reward -= penalty
        
        self.prev_action = action.copy()
        return obs, reward, terminated, truncated, info

    def reset(self, **kwargs):
        self.prev_action = None
        return self.env.reset(**kwargs)

class ControlCostWrapper(gym.Wrapper):
    def __init__(self, env, penalty_coef=0.02, force_margin=5.0):
        super().__init__(env)
        self.penalty_coef = penalty_coef  # 惩罚系数
        self.force_margin = force_margin
        
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        # 获取力矩惩罚
        if hasattr(self.env.unwrapped, 'robot'):
            forces = self.env.unwrapped.robot.actuator_forces()
            # 超过force_margin的力矩被惩罚
            force_penalty = np.maximum(0, forces - self.force_margin).mean()
            reward -= self.penalty_coef * force_penalty
        
        return obs, reward, terminated, truncated, info
    
    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

class HardBipedalWalkerWrapper(gym.Wrapper):
    def __init__(
        self,
        env,
        penalty_coef=0.001,
        milestone_dist=1.0,
        milestone_reward=10.0,
        failure_penalty=30.0,
        mask_lidar=True,
    ):
        super().__init__(env)
        self.penalty_coef = penalty_coef
        self.milestone_dist = milestone_dist
        self.milestone_reward = milestone_reward
        self.failure_penalty = failure_penalty
        self.mask_lidar = mask_lidar

        self.prev_action = None
        self.last_milestone_x = 0.0
        self.max_x = 0.0

    def _process_obs(self, obs):
        if self.mask_lidar:
            obs = obs.copy()
            obs[8:22] = 0.0
        return obs

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        obs = self._process_obs(obs)

        current_x = self.env.unwrapped.hull.position.x
        custom_reward = 0.0

        while current_x > self.last_milestone_x + self.milestone_dist:
            self.last_milestone_x += self.milestone_dist
            custom_reward += self.milestone_reward

        if terminated:
            custom_reward -= self.failure_penalty

        self.max_x = max(self.max_x, current_x)

        # --- 动作平滑惩罚 ---
        if self.prev_action is not None:
            penalty = self.penalty_coef * np.sum((action - self.prev_action) ** 2)
            custom_reward -= penalty

        self.prev_action = action.copy()

        return obs, custom_reward, terminated, truncated, info

    def reset(self, **kwargs):
        self.prev_action = None
        self.last_milestone_x = 0.0
        self.max_x = 0.0
        obs, info = self.env.reset(**kwargs)
        return self._process_obs(obs), info
    
class SparseBipedalWalkerWrapper(gym.Wrapper):
    """
    Truly sparse-reward version of BipedalWalker.

    Reward design:
    - +success_reward only if reaching goal_x
    - -failure_penalty if falling
    - Optional small action smoothness penalty

    No dense progress reward.
    No milestone reward.
    """

    def __init__(
        self,
        env,
        goal_x=30.0,
        success_reward=100.0,
        failure_penalty=10.0,
        penalty_coef=0.001,
        mask_lidar=False,
    ):
        super().__init__(env)

        self.goal_x = goal_x
        self.success_reward = success_reward
        self.failure_penalty = failure_penalty
        self.penalty_coef = penalty_coef
        self.mask_lidar = mask_lidar

        self.prev_action = None
        self.success = False

    def _process_obs(self, obs):
        if self.mask_lidar:
            obs = obs.copy()
            obs[8:22] = 0.0
        return obs

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        obs = self._process_obs(obs)

        current_x = self.env.unwrapped.hull.position.x

        custom_reward = 0.0

        # --- success reward ---
        if (not self.success) and current_x >= self.goal_x:
            custom_reward += self.success_reward
            self.success = True

            # terminate immediately after success
            terminated = True
            info["success"] = True
            info["dist"] = current_x

        # --- failure penalty ---
        elif terminated:
            custom_reward -= self.failure_penalty
            info["success"] = False
            info["dist"] = current_x

        # --- action smoothness penalty ---
        if self.prev_action is not None:
            penalty = self.penalty_coef * np.sum(
                (action - self.prev_action) ** 2
            )
            custom_reward -= penalty

        self.prev_action = action.copy()

        return obs, custom_reward, terminated, truncated, info

    def reset(self, **kwargs):
        self.prev_action = None
        self.success = False

        obs, info = self.env.reset(**kwargs)

        return self._process_obs(obs), info