"""
GRPO with a KL penalty to a frozen reference policy (GRPO+KL ablation).

Standard GRPO (DeepSeekMath) adds a KL penalty beta * KL(pi_theta || pi_ref) to the
clipped surrogate loss, where pi_ref is the reference policy. Here the reference is a
frozen copy of the policy taken at the first update (i.e. the loaded pretrained
baseline policy). This variant keeps the group-relative advantage machinery from
utils.grpo.GRPO intact and only adds the KL term, so it can be used as an ablation to
disentangle the effect of group-relative advantages from the absence of the KL penalty.

Loss:  L = -E[min(rho*A, clip(rho, 1-eps, 1+eps)*A)] + kl_coef * KL(pi_theta || pi_ref)

Usage (via finetune_hb.py):
    python finetune_hb.py --algo grpo --kl_coef 0.01 --env_name h1hand-door-v0 \
        --exp_name ft_grpokl_v1 --seed 42
"""

import copy
from typing import Any, List, Optional

import numpy as np
import torch as th
from gymnasium import spaces
from torch.nn import functional as F

from stable_baselines3.common.buffers import RolloutBuffer
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.utils import explained_variance
from stable_baselines3.common.vec_env import VecEnv

from utils.grpo import GRPO


class GRPOKL(GRPO):
    """
    Group Relative Policy Optimization with a KL penalty to the reference policy.

    Args:
        policy: Policy model (MlpPolicy, ...)
        env: Environment
        kl_coef: Coefficient of the KL penalty term (beta). 0 disables the KL term
            and behaves exactly like utils.grpo.GRPO.
        **kwargs: Standard PPO arguments
    """

    def __init__(
        self,
        policy: Any,
        env: VecEnv,
        kl_coef: float = 0.0,
        **kwargs,
    ):
        self.kl_coef = float(kl_coef)
        # Frozen reference policy (pretrained baseline), lazily created at first update
        self._ref_policy: Optional[Any] = None
        super().__init__(policy=policy, env=env, **kwargs)

    def _ensure_ref_policy(self) -> None:
        """Freeze a copy of the current (pretrained) policy as the KL reference."""
        if self._ref_policy is None:
            self._ref_policy = copy.deepcopy(self.policy)
            self._ref_policy.set_training_mode(False)
            # The reference only needs the pi network, not its optimizer
            self._ref_policy.optimizer = None
            self._ref_policy = self._ref_policy.to(self.device)

    def train(self) -> None:
        """
        Update policy using the currently gathered rollout buffer.

        Copy of stable_baselines3 2.x PPO.train() with an extra KL penalty
        term kl_coef * KL(pi_theta || pi_ref) added to the loss.
        """
        self._ensure_ref_policy()

        # Switch to train mode (this affects batch norm / dropout)
        self.policy.set_training_mode(True)
        # Update optimizer learning rate
        self._update_learning_rate(self.policy.optimizer)
        # Compute current clip range
        clip_range = self.clip_range(self._current_progress_remaining)  # type: ignore[operator]
        # Optional: clip range for the value function
        if self.clip_range_vf is not None:
            clip_range_vf = self.clip_range_vf(self._current_progress_remaining)  # type: ignore[operator]

        entropy_losses = []
        pg_losses, value_losses = [], []
        clip_fractions = []
        kl_losses = []

        continue_training = True
        # train for n_epochs epochs
        for epoch in range(self.n_epochs):
            approx_kl_divs = []
            # Do a complete pass on the rollout buffer
            for rollout_data in self.rollout_buffer.get(self.batch_size):
                actions = rollout_data.actions
                if isinstance(self.action_space, spaces.Discrete):
                    # Convert discrete action from float to long
                    actions = rollout_data.actions.long().flatten()

                # Re-sample the noise matrix because the log_std has changed
                if self.use_sde:
                    self.policy.reset_noise(self.batch_size)

                # KL to the frozen reference policy (no gradient through it)
                if self.kl_coef > 0:
                    with th.no_grad():
                        ref_dist = self._ref_policy.get_distribution(rollout_data.observations)

                values, log_prob, entropy = self.policy.evaluate_actions(rollout_data.observations, actions)
                values = values.flatten()
                # Normalize advantage
                advantages = rollout_data.advantages
                # Normalization does not make sense if mini batchsize == 1, see GH issue #325
                if self.normalize_advantage and len(advantages) > 1:
                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                # ratio between old and new policy, should be one at the first iteration
                ratio = th.exp(log_prob - rollout_data.old_log_prob)

                # clipped surrogate loss
                policy_loss_1 = advantages * ratio
                policy_loss_2 = advantages * th.clamp(ratio, 1 - clip_range, 1 + clip_range)
                policy_loss = -th.min(policy_loss_1, policy_loss_2).mean()

                # Logging
                pg_losses.append(policy_loss.item())
                clip_fraction = th.mean((th.abs(ratio - 1) > clip_range).float()).item()
                clip_fractions.append(clip_fraction)

                if self.clip_range_vf is None:
                    # No clipping
                    values_pred = values
                else:
                    # Clip the difference between old and new value
                    # NOTE: this depends on the reward scaling
                    values_pred = rollout_data.old_values + th.clamp(
                        values - rollout_data.old_values, -clip_range_vf, clip_range_vf
                    )
                # Value loss using the TD(gae_lambda) target
                value_loss = F.mse_loss(rollout_data.returns, values_pred)
                value_losses.append(value_loss.item())

                # Entropy loss favor exploration
                if entropy is None:
                    # Approximate entropy when no analytical form
                    entropy_loss = -th.mean(-log_prob)
                else:
                    entropy_loss = -th.mean(entropy)

                entropy_losses.append(entropy_loss.item())

                # KL penalty term: KL(pi_theta || pi_ref), summed over action dims
                if self.kl_coef > 0:
                    cur_dist = self.policy.get_distribution(rollout_data.observations)
                    kl_div = th.distributions.kl_divergence(
                        cur_dist.distribution, ref_dist.distribution
                    )
                    kl_loss = kl_div.sum(dim=-1).mean()
                    kl_losses.append(kl_loss.item())
                else:
                    kl_loss = th.zeros((), device=self.device)

                loss = (
                    policy_loss
                    + self.ent_coef * entropy_loss
                    + self.vf_coef * value_loss
                    + self.kl_coef * kl_loss
                )

                # Calculate approximate form of reverse KL Divergence for early stopping
                # see issue #417: https://github.com/DLR-RM/stable-baselines3/issues/417
                # and discussion in PR #419: https://github.com/DLR-RM/stable-baselines3/pull/419
                with th.no_grad():
                    log_ratio = log_prob - rollout_data.old_log_prob
                    approx_kl_div = th.mean((th.exp(log_ratio) - 1) - log_ratio).cpu().numpy()
                    approx_kl_divs.append(approx_kl_div)

                if self.target_kl is not None and approx_kl_div > 1.5 * self.target_kl:
                    continue_training = False
                    if self.verbose >= 1:
                        print(f"Early stopping at step {epoch} due to reaching max kl: {approx_kl_div:.2f}")
                    break

                # Optimization step
                self.policy.optimizer.zero_grad()
                loss.backward()
                # Clip grad norm
                th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.policy.optimizer.step()

            self._n_updates += 1
            if not continue_training:
                break

        explained_var = explained_variance(self.rollout_buffer.values.flatten(), self.rollout_buffer.returns.flatten())

        # Logs
        self.logger.record("train/entropy_loss", np.mean(entropy_losses))
        self.logger.record("train/policy_gradient_loss", np.mean(pg_losses))
        self.logger.record("train/value_loss", np.mean(value_losses))
        self.logger.record("train/approx_kl", np.mean(approx_kl_divs))
        self.logger.record("train/clip_fraction", np.mean(clip_fractions))
        if self.kl_coef > 0:
            self.logger.record("train/kl_penalty", np.mean(kl_losses))
        self.logger.record("train/loss", loss.item())
        self.logger.record("train/explained_variance", explained_var)
        if hasattr(self.policy, "log_std"):
            self.logger.record("train/std", th.exp(self.policy.log_std).mean().item())

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/clip_range", clip_range)
        if self.clip_range_vf is not None:
            self.logger.record("train/clip_range_vf", clip_range_vf)

    def _excluded_save_params(self) -> List[str]:
        """Exclude non-serializable components from saved params."""
        excluded = super()._excluded_save_params()
        # The frozen reference policy is re-created lazily from the loaded checkpoint
        return excluded + ["_ref_policy"]
