"""
Group Relative Policy Optimization (GRPO)
Based on PPO implementation from stable-baselines3
Support for external state/action grouping functions (e.g., clustering-based)
"""

from typing import Callable, Optional, Any, List
import numpy as np
import torch as th
import torch.nn as nn

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
        feature_extractor: Optional nn.Module to extract features from observations.
            If None, uses raw observations directly (identity).
        grouping_strategy: Optional callable that takes features (numpy array of shape 
            (n_samples, feature_dim)) and returns group IDs (numpy array of shape (n_samples,)).
            If None, all samples are placed in group 0.
        **kwargs: Standard PPO arguments
    """
    
    def __init__(
        self,
        policy: Any,
        env: VecEnv,
        feature_extractor: Optional[nn.Module] = None,
        grouping_strategy: Optional[Callable] = None,
        **kwargs,
    ):
        # Store feature extractor and grouping strategy
        self.feature_extractor = feature_extractor if feature_extractor is not None else nn.Identity()
        self.grouping_strategy = grouping_strategy
        
        # Initialize parent PPO
        super().__init__(policy=policy, env=env, **kwargs)
        
        # Move feature extractor to same device as policy
        self.feature_extractor = self.feature_extractor.to(self.device)
        
        # CRITICAL: Disable critic network training
        self.vf_coef = 0.0
        
        # CRITICAL: Disable advantage normalization (GRPO uses its own)
        self.normalize_advantage = False
    
    def _extract_features(self, obs: th.Tensor) -> th.Tensor:
        """
        Extract features from observations using the feature extractor.
        
        Args:
            obs: Observation tensor of shape (n_samples, obs_dim)
        
        Returns:
            Features tensor of shape (n_samples, feature_dim)
        """
        with th.no_grad():
            features = self.feature_extractor(obs)
        return features
    
    def _get_group_ids(self, features: th.Tensor) -> np.ndarray:
        """
        Compute group IDs for a batch of features.
        
        Args:
            features: Array of shape (n_samples, feature_dim)
        
        Returns:
            Array of shape (n_samples,) with group IDs
        """
        if self.grouping_strategy is None:
            return np.zeros(len(features), dtype=np.int64)
        
        group_ids = self.grouping_strategy(features)
        return np.asarray(group_ids, dtype=np.int64)
    
    def _compute_group_advantages(
        self,
        returns: np.ndarray,
        group_ids: np.ndarray,
        eps: float = 1e-8,
    ) -> np.ndarray:
        """
        Compute group-relative advantages with fallback to global normalization.
        
        For each unique group:
        - If group has >= 2 samples: advantage = (r - mean_r) / (std_r + eps)
        - If group has 1 sample: advantage = (r - global_mean) / (global_std + eps)
        - If group has 0 samples: skip
        
        Args:
            returns: Array of shape (n_samples,) - discounted returns
            group_ids: Array of shape (n_samples,) - group ID for each sample
            eps: Small constant for numerical stability
        
        Returns:
            advantages: Array of shape (n_samples,)
        """
        n_samples = len(returns)
        advantages = np.zeros(n_samples, dtype=np.float32)
        
        # Global statistics for fallback
        global_mean = returns.mean()
        global_std = returns.std() + eps
        
        # Get unique groups
        unique_groups = np.unique(group_ids)
        
        # Compute advantages for each group
        for group in unique_groups:
            indices = np.where(group_ids == group)[0]
            if len(indices) == 0:
                continue
            elif len(indices) == 1:
                # Single sample: fallback to global normalization
                advantages[indices[0]] = (returns[indices[0]] - global_mean) / global_std
            else:
                # Multiple samples: group-relative normalization
                group_returns = returns[indices]
                mean_r = group_returns.mean()
                std_r = group_returns.std() + eps
                advantages[indices] = (group_returns - mean_r) / std_r
        
        return advantages
    
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
        
        # Reshape returns to flat array for group computation
        # Shape: (n_rollout_steps * n_envs,)
        flat_returns = returns.flatten()
        
        # Get observations from buffer
        n_envs = env.num_envs
        n_steps = n_rollout_steps
        n_total = n_steps * n_envs
        
        # Extract observations from buffer
        obs_flat = rollout_buffer.observations.reshape(n_total, -1)
        
        # Extract features and compute group IDs
        obs_tensor = th.as_tensor(obs_flat, dtype=th.float32).to(self.device)
        features = self._extract_features(obs_tensor)
        
        # Compute group IDs (shape: n_total,)
        group_ids = self._get_group_ids(features)
        
        # Compute group-relative advantages using only group IDs
        advantages = self._compute_group_advantages(flat_returns, group_ids)
        
        # Reshape and store in buffer
        rollout_buffer.advantages = advantages.reshape(n_steps, n_envs).copy()
        rollout_buffer.returns = flat_returns.reshape(n_steps, n_envs).copy()
        
        # Do NOT call rollout_buffer.compute_returns_and_advantage()
        # That would overwrite our advantages with GAE
        
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
    
    def _excluded_save_params(self) -> List[str]:
        """Exclude non-serializable components from saved params."""
        excluded = super()._excluded_save_params()
        # Exclude grouping_strategy if they can't be pickled
        return excluded + ["grouping_strategy"]