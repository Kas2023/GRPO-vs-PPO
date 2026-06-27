"""
Group Relative Policy Optimization (GRPO)
Based on PPO implementation from stable-baselines3
"""

from typing import Any
import numpy as np
import torch as th

from stable_baselines3 import PPO
from stable_baselines3.common.buffers import RolloutBuffer
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import VecEnv


class GRPO(PPO):
    """
    Group Relative Policy Optimization (GRPO)
    
    Args:
        policy: Policy model (MlpPolicy, CnnPolicy, etc.)
        env: Environment
        **kwargs: Standard PPO arguments
    """
    
    def __init__(
        self,
        policy: Any,
        env: VecEnv,
        **kwargs,
    ):
        super().__init__(policy=policy, env=env, **kwargs)

        # Disable critic training
        self.vf_coef = 0.0

        # Disable PPO advantage normalization
        self.normalize_advantage = False
    
    def _compute_trajectory_advantages(
        self,
        trajectory_returns: np.ndarray,
        eps: float = 1e-8,
    ) -> np.ndarray:
        """
        Compute trajectory-level normalized advantages.

        Args:
            trajectory_returns: shape (n_envs,)

        Returns:
            advantages: shape (n_envs,)
        """
        mean_r = trajectory_returns.mean()
        std_r = trajectory_returns.std() + eps

        advantages = (trajectory_returns - mean_r) / std_r
        return advantages.astype(np.float32)

    def collect_rollouts(
        self,
        env: VecEnv,
        callback: BaseCallback,
        rollout_buffer: RolloutBuffer,
        n_rollout_steps: int,
    ) -> bool:
        """
        Collect rollouts and compute group-relative advantages.
        
        Args:
            env: Vectorized environment
            callback: Callback for logging/early stopping
            rollout_buffer: Buffer to store transitions
            n_rollout_steps: Number of steps to collect
        
        Returns:
            True if training should continue, False otherwise
        """
        assert self._last_obs is not None, "No previous observation was provided"
        rollout_buffer.reset()
        self.policy.set_training_mode(False)
        callback.on_rollout_start()
        
        # Storage for step-by-step data
        step_rewards = np.zeros((n_rollout_steps, env.num_envs), dtype=np.float32)
        step_dones = np.zeros((n_rollout_steps, env.num_envs), dtype=bool)
        
        for step in range(n_rollout_steps):
            if callback.on_step() is False:
                return False
            
            # Sample action from policy
            with th.no_grad():
                obs_tensor = th.as_tensor(self._last_obs).to(self.device)
                actions, values, log_probs = self.policy(obs_tensor)
            
            actions_np = actions.cpu().numpy()
            
            # Step environment
            new_obs, rewards, dones, infos = env.step(actions_np)
            self.num_timesteps += env.num_envs
            
            # Store rewards and dones for return calculation
            step_rewards[step] = rewards.copy()
            step_dones[step] = dones.copy()
            
            # Add to buffer (rewards will be overwritten later)
            rollout_buffer.add(
                self._last_obs,
                actions_np,
                rewards,
                dones,
                values,
                log_probs,
            )
            
            self._last_obs = new_obs
            self._last_episode_starts = dones
        
        # Compute discounted returns for each step
        # Returns are computed backwards to handle episode boundaries correctly
        returns = np.zeros_like(step_rewards)
        
        for env_idx in range(env.num_envs):
            R = 0.0
            for t in reversed(range(n_rollout_steps)):
                if step_dones[t, env_idx]:
                    R = 0.0
                R = step_rewards[t, env_idx] + self.gamma * R
                returns[t, env_idx] = R
        
        n_envs = env.num_envs
        n_steps = n_rollout_steps

        # Use episodic return proxy:
        # total discounted return of each environment rollout
        trajectory_returns = returns[0]

        # Compute trajectory-level relative advantages
        trajectory_advantages = self._compute_trajectory_advantages(
            trajectory_returns
        )

        # Expand trajectory advantage to all timesteps
        advantages = np.tile(
            trajectory_advantages,
            (n_steps, 1)
        )

        # Store into rollout buffer
        rollout_buffer.advantages = advantages.copy()
        rollout_buffer.returns = returns.copy()
                
        callback.on_rollout_end()
        return True
    
    def train(self) -> None:
        """
        Override train to ensure no value loss is computed.
        
        Since we set vf_coef=0, parent's train() will work correctly.
        The value loss will be multiplied by 0, and advantage normalization
        is disabled via normalize_advantage=False.
        """
        super().train()