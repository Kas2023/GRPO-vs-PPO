"""
Fine-tuning PPO vs GRPO on Reach task with Sparse Success Reward.

Reproduces the Door task's Phase 2 Sparse Reward condition on the Reach task:
  - Replaces the original dense reward (healthy_reward + motion_penalty
    + reward_close + reward_success) with a terminal-only binary signal:
    reward = 1.0 at episode end if min_hand_dist < 0.05, else 0.0.
    All intermediate steps receive 0. This matches Door Phase 2's
    sparse level (max 1 reward per episode).

The baseline reach model already achieves ~100% success rate.  This
experiment tests whether PPO and GRPO can *maintain* that performance
when the reward is reduced to a sparse binary success signal.

Key design:
  1. Loads pretrained reach weights (TorchModel → SB3 policy), expanding
     from 55D→256→256→19 to the full 155D→256→256→61 architecture.
     Unobserved dimensions get zero-initialized weights; hand-joint action
     rows get zero weights+bias (→ tanh(0)=0 → neutral joint position).
  2. VecNormalize stats initialised from reach data, expanded to 155D.
  3. Full 155D observation, 61D action — NO observation/action wrappers.
  4. ReachSparseWrapper: replaces dense reward with binary success signal,
     tracks success, hand_dist, and action smoothness.

Usage:
  python finetune_reach.py --algo ppo  --exp_name reach_sparse_ppo  --seed 42
  python finetune_reach.py --algo grpo --exp_name reach_sparse_grpo --seed 42
"""

import argparse
import os
import sys
import numpy as np
import torch as th
import gymnasium as gym

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.running_mean_std import RunningMeanStd

import humanoid_bench
from humanoid_bench.wrappers import get_body_idxs
from utils.grpo import GRPO

# ──────────────────────────────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────────────────────────────
REACH_MODEL_DIR = "humanoid-bench/data/reach_one_hand"
REACH_MODEL_PATH = os.path.join(REACH_MODEL_DIR, "torch_model.pt")
REACH_MEAN_PATH = os.path.join(REACH_MODEL_DIR, "mean.npy")
REACH_VAR_PATH = os.path.join(REACH_MODEL_DIR, "var.npy")

# ──────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--algo", type=str, default="ppo", choices=["ppo", "grpo"])
parser.add_argument("--exp_name", type=str, required=True)
parser.add_argument("--seed", type=int, required=True)
parser.add_argument("--num_envs", default=4, type=int)
parser.add_argument("--learning_rate", default=3e-5, type=float)
parser.add_argument("--max_steps", default=2_000_000, type=int)
parser.add_argument("--eval_freq", default=50_000, type=int)
parser.add_argument("--n_eval_episodes", default=20, type=int)
parser.add_argument("--wandb_entity", default="wlx-k-s-2003-ucl", type=str)
ARGS = parser.parse_args()


# ──────────────────────────────────────────────────────────────────────
# Sparse reward wrapper: replaces dense reward with binary success signal
# ──────────────────────────────────────────────────────────────────────

class ReachSparseWrapper(gym.Wrapper):
    """Replaces the original dense reach reward with a sparse binary success
    signal: reward = 1.0 if hand_dist < 0.05 else 0.0 at each step.

    Also applies offset-centering to the observation so that position,
    left_hand, and target coordinates are relative to the robot's current
    body position (matching the pretrained reach model's input convention).
    The output is still 155D — only the values of body-position, left_hand,
    and target dimensions are shifted.

    Tracks per-episode metrics: success, min_hand_dist, and action
    smoothness (on the full 61D action).
    """

    def __init__(self, env):
        super().__init__(env)
        self._task = env.unwrapped.task
        model = env.unwrapped.model
        self._body_idxs, self._body_vel_idxs = get_body_idxs(model)
        self._robot_dof = env.unwrapped.robot.dof

        # episode metrics
        self._episode_success = False
        self._prev_action = None
        self._episode_action_sqdiff = 0.0
        self._min_hand_dist = float("inf")

    def _center_obs(self, obs: np.ndarray) -> np.ndarray:
        """Apply offset centering so the model sees relative, not absolute,
        coordinates — matching the pretrained reach model's input convention.

        The 155D layout (from Reach.get_obs):
            obs[0:robot_dof]          = position  (75D)
            obs[robot_dof:2*robot_dof-1] = velocity  (74D)
            obs[2*robot_dof-1:2*robot_dof+2] = left_hand (3D)
            obs[2*robot_dof+2:]       = target    (3D)

        We subtract (body_pos.x, body_pos.y, 0) from the first body-part
        position, left_hand, and target — same logic as SingleReachWrapper.
        """
        robot_dof = self._robot_dof
        position = obs[:robot_dof].copy()
        velocity = obs[robot_dof:robot_dof * 2 - 1]
        left_hand = obs[robot_dof * 2 - 1:robot_dof * 2 + 2].copy()
        target = obs[robot_dof * 2 + 2:].copy()

        body_pos = position[self._body_idxs]
        offset = np.array([body_pos[0], body_pos[1], 0.0])

        body_pos[:3] -= offset
        position[self._body_idxs] = body_pos
        left_hand -= offset
        target -= offset

        return np.concatenate([position, velocity, left_hand, target])

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._episode_success = False
        self._prev_action = None
        self._episode_action_sqdiff = 0.0
        self._min_hand_dist = float("inf")
        return self._center_obs(obs), info

    def step(self, action):
        obs, _reward, terminated, truncated, info = self.env.step(action)

        # Compute hand distance from raw env internals
        hand_dist = float(np.sqrt(
            np.square(self._task.robot.left_hand_position() - self._task.goal).sum()
        ))

        # Terminal-only sparse reward (aligned with Door Phase 2):
        # reward = 1.0 at episode end if hand ever reached the target
        # (min_hand_dist < 0.05), else 0.0.  All intermediate steps are 0.
        # This matches the "success" definition and makes the sparse level
        # comparable to Door Phase 2 (max 1 reward per episode).
        if hand_dist < self._min_hand_dist:
            self._min_hand_dist = hand_dist
        if hand_dist < 0.05:
            self._episode_success = True

        reward = 0.0  # intermediate steps: no reward

        # Action smoothness (full 61D action)
        if self._prev_action is not None:
            self._episode_action_sqdiff += float(
                np.sum(np.square(action - self._prev_action))
            )
        self._prev_action = action.copy()

        info["hand_dist"] = hand_dist

        if terminated or truncated:
            reward = 1.0 if self._episode_success else 0.0
            if self._episode_success:
                info["success"] = 1
            info["action_smoothness"] = self._episode_action_sqdiff
            info["min_hand_dist"] = self._min_hand_dist

        return self._center_obs(obs), reward, terminated, truncated, info


# ──────────────────────────────────────────────────────────────────────
# Environment factory
# ──────────────────────────────────────────────────────────────────────

def make_env(rank: int, seed: int = 0):
    def _init():
        env = gym.make("h1hand-reach-v0", render_mode="rgb_array")
        env = ReachSparseWrapper(env)    # reward override + metrics
        # gym.make already applies TimeLimit(max_episode_steps=1000) from
        # Task.max_episode_steps; do NOT add a second one.
        env = Monitor(env, f"logs/h1hand-reach-v0/{ARGS.exp_name}/train/env_{rank}")
        env.reset(seed=ARGS.seed + rank)
        return env
    return _init


# ──────────────────────────────────────────────────────────────────────
# Dimension mapping: 55D→155D (obs) and 19→61 (action)
# ──────────────────────────────────────────────────────────────────────

def get_obs_mapping(env) -> list:
    """Compute the 55D→155D observation index mapping.

    The pretrained reach model sees 55D:
        position[body_idxs][2:] (24D) + velocity[body_vel_idxs] (25D)
        + left_hand (3D) + target (3D)

    The full observation is 155D:
        position (75D) + velocity (74D) + left_hand (3D) + target (3D)

    Returns a list `m` of length 55 where m[j] is the 155D index
    corresponding to the j-th dimension of the 55D observation.
    """
    model = env.unwrapped.model
    body_idxs, body_vel_idxs = get_body_idxs(model)
    robot_dof = env.unwrapped.robot.dof

    pos_start = 0
    vel_start = robot_dof
    lh_start = robot_dof * 2 - 1      # 3 dims
    target_start = robot_dof * 2 + 2   # 3 dims

    mapping = []
    # position[body_idxs][2:] → within position (75D)
    for idx in body_idxs[2:]:
        mapping.append(pos_start + idx)
    # velocity[body_vel_idxs] → within velocity (74D)
    for idx in body_vel_idxs:
        mapping.append(vel_start + idx)
    # left_hand (3D)
    for i in range(3):
        mapping.append(lh_start + i)
    # target (3D)
    for i in range(3):
        mapping.append(target_start + i)

    assert len(mapping) == 55, f"Expected 55 obs dims, got {len(mapping)}"
    return mapping


def get_action_mapping(env) -> list:
    """Compute the 19→61 action index mapping.

    The pretrained reach model outputs 19D body-joint actions.
    The full action space is 61D (body + hand joints).

    Returns a list `m` of length 19 where m[j] is the 61D index
    corresponding to the j-th dimension of the 19D action.
    """
    if env.unwrapped.model.nu > 19:
        act_idxs = list(range(15)) + list(range(16, 20))  # 19 body joints
    else:
        act_idxs = list(range(19))
    return act_idxs


# ──────────────────────────────────────────────────────────────────────
# Weight loading: TorchModel (55→19) → SB3 policy (155→61)
# ──────────────────────────────────────────────────────────────────────

def load_reach_weights_expanded(model, pt_path: str, mean_path: str, var_path: str,
                                 obs_mapping: list, act_mapping: list):
    """Load pretrained reach weights into an SB3 policy with expanded dims.

    Architecture mapping (with expansion):
        TorchModel.dense1 (256×55) → mlp_extractor.policy_net.0 (256×155)
            - 55 columns scattered to 155 according to obs_mapping
            - unmapped columns remain zero
        TorchModel.dense2 (256×256) → mlp_extractor.policy_net.2 (256×256)
            - direct copy (unchanged)
        TorchModel.dense3 (19×256) → action_net (61×256)
            - 19 rows scattered to 61 according to act_mapping
            - unmapped rows (hand joints) remain zero → tanh(0)=0 → neutral
        log_std: 19 → 61 (scattered; unmapped dims init to -1.0)

    Value network and unmapped dimensions remain randomly initialised.
    VecNormalize obs stats are expanded from 55D to 155D (unmapped dims:
    mean=0, var=1).
    """
    pt_weights = th.load(pt_path, map_location="cpu")
    policy_state = model.policy.state_dict()

    full_obs_dim = len(policy_state["mlp_extractor.policy_net.0.weight"][0])   # 155
    full_act_dim = len(policy_state["action_net.bias"])                          # 61

    # ── dense1.weight: (256, 55) → (256, 155) ──
    old_w1 = pt_weights["dense1.weight"]  # (256, 55)
    new_w1 = th.zeros(256, full_obs_dim)
    for j, target_col in enumerate(obs_mapping):
        new_w1[:, target_col] = old_w1[:, j]
    policy_state["mlp_extractor.policy_net.0.weight"] = new_w1
    # bias unchanged: (256,) → (256,)
    policy_state["mlp_extractor.policy_net.0.bias"] = pt_weights["dense1.bias"]

    # ── dense2: (256, 256) → (256, 256) — direct copy ──
    policy_state["mlp_extractor.policy_net.2.weight"] = pt_weights["dense2.weight"]
    policy_state["mlp_extractor.policy_net.2.bias"] = pt_weights["dense2.bias"]

    # ── dense3.weight: (19, 256) → (61, 256) ──
    old_w3 = pt_weights["dense3.weight"]  # (19, 256)
    new_w3 = th.zeros(full_act_dim, 256)
    for j, target_row in enumerate(act_mapping):
        new_w3[target_row] = old_w3[j]
    policy_state["action_net.weight"] = new_w3

    # ── dense3.bias: (19,) → (61,) ──
    old_b3 = pt_weights["dense3.bias"]  # (19,)
    new_b3 = th.zeros(full_act_dim)
    for j, target_row in enumerate(act_mapping):
        new_b3[target_row] = old_b3[j]
    policy_state["action_net.bias"] = new_b3

    # ── log_std: (61,) — scatter body-joint values, hand-joint init -1.0 ──
    old_log_std = policy_state["log_std"]  # freshly initialised (61,)
    new_log_std = th.full((full_act_dim,), -1.0)
    for target_row in act_mapping:
        new_log_std[target_row] = old_log_std[target_row]
    policy_state["log_std"] = new_log_std

    model.policy.load_state_dict(policy_state)
    print(f"  Expanded policy weights: 55→{full_obs_dim} obs, 19→{full_act_dim} act")

    # ── VecNormalize obs stats: 55D → 155D ──
    pt_mean_55 = np.load(mean_path)[0].astype(np.float64)  # (55,)
    pt_var_55 = np.load(var_path)[0].astype(np.float64)    # (55,)

    pt_mean_full = np.zeros(full_obs_dim, dtype=np.float64)
    pt_var_full = np.ones(full_obs_dim, dtype=np.float64)  # default var=1
    for j, target_col in enumerate(obs_mapping):
        pt_mean_full[target_col] = pt_mean_55[j]
        pt_var_full[target_col] = pt_var_55[j]

    train_env = model.get_vec_normalize_env()
    if train_env is not None:
        train_env.obs_rms.mean = pt_mean_full
        train_env.obs_rms.var = pt_var_full
        train_env.obs_rms.count = 1e4  # high count → slow adaptation
        print(f"  Initialised VecNormalize obs stats (55→{full_obs_dim})")


# ──────────────────────────────────────────────────────────────────────
# Callbacks
# ──────────────────────────────────────────────────────────────────────

class EpisodeLogCallback(BaseCallback):
    """Log per-episode return, length, success rate, and action smoothness."""

    def __init__(self):
        super().__init__()
        self.ep_returns = []
        self.ep_lengths = []
        self.ep_successes = []
        self.ep_action_smoothness = []
        self.ep_min_hand_dist = []

    def _on_step(self) -> bool:
        infos = self.locals["infos"]
        for info in infos:
            if "episode" in info:
                self.ep_returns.append(info["episode"]["r"])
                self.ep_lengths.append(info["episode"]["l"])
                self.ep_successes.append(info.get("success", 0))
                self.ep_action_smoothness.append(
                    info.get("action_smoothness", 0.0))
                self.ep_min_hand_dist.append(
                    info.get("min_hand_dist", float("inf")))
        return True

    def _on_rollout_end(self) -> None:
        if self.ep_returns:
            self.logger.record("results/mean_return", np.mean(self.ep_returns))
            self.logger.record("results/mean_length", np.mean(self.ep_lengths))
            self.logger.record("results/success_rate", np.mean(self.ep_successes))
        if self.ep_action_smoothness:
            self.logger.record("behavior/action_smoothness",
                               np.mean(self.ep_action_smoothness))
        if self.ep_min_hand_dist:
            self.logger.record("results/min_hand_dist",
                               np.mean(self.ep_min_hand_dist))
        self.ep_returns = []
        self.ep_lengths = []
        self.ep_successes = []
        self.ep_action_smoothness = []
        self.ep_min_hand_dist = []


class PeriodicEvalCallback(BaseCallback):
    """Periodic evaluation with dual-criterion best-model saving."""

    def __init__(self, eval_env, model_save_dir: str,
                 eval_freq: int, n_eval_episodes: int):
        super().__init__()
        self.eval_env = eval_env
        self.model_save_dir = model_save_dir
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes
        self.best_eval_success_rate = -1.0
        self.best_eval_mean_reward = -float("inf")

    def _on_step(self) -> bool:
        if self.n_calls % self.eval_freq != 0:
            return True

        train_vec_norm = self.model.get_vec_normalize_env()
        if train_vec_norm is not None:
            self.eval_env.obs_rms = train_vec_norm.obs_rms
            self.eval_env.ret_rms = train_vec_norm.ret_rms

        episode_rewards = []
        episode_successes = []
        episode_lengths = []
        episode_min_hand_dists = []

        for _ in range(self.n_eval_episodes):
            obs = self.eval_env.reset()
            done = False
            ep_rew = 0.0
            ep_len = 0
            ep_success = False
            ep_min_dist = float("inf")

            while not done:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, rewards, dones, infos = self.eval_env.step(action)
                ep_rew += float(rewards[0])
                ep_len += 1
                if infos[0].get("success", 0) == 1:
                    ep_success = True
                d = infos[0].get("min_hand_dist", float("inf"))
                if d < ep_min_dist:
                    ep_min_dist = d
                done = bool(dones[0])

            episode_rewards.append(ep_rew)
            episode_successes.append(1.0 if ep_success else 0.0)
            episode_lengths.append(ep_len)
            episode_min_hand_dists.append(ep_min_dist)

        mean_reward = float(np.mean(episode_rewards))
        success_rate = float(np.mean(episode_successes))
        mean_length = float(np.mean(episode_lengths))
        mean_min_dist = float(np.mean(episode_min_hand_dists))

        self.logger.record("eval/mean_reward", mean_reward)
        self.logger.record("eval/success_rate", success_rate)
        self.logger.record("eval/mean_ep_length", mean_length)
        self.logger.record("eval/min_hand_dist", mean_min_dist)

        steps = self.num_timesteps

        # Periodic snapshot
        self.model.save(os.path.join(self.model_save_dir, f"model_{steps}_steps"))
        if train_vec_norm is not None:
            train_vec_norm.save(os.path.join(
                self.model_save_dir, f"vecnormalize_{steps}_steps.pkl"))

        # Best by success
        if success_rate >= self.best_eval_success_rate:
            self.best_eval_success_rate = success_rate
            self.model.save(os.path.join(self.model_save_dir, "best_by_success"))
            if train_vec_norm is not None:
                train_vec_norm.save(os.path.join(
                    self.model_save_dir, "best_by_success_vecnormalize.pkl"))

        # Best by reward
        if mean_reward >= self.best_eval_mean_reward:
            self.best_eval_mean_reward = mean_reward
            self.model.save(os.path.join(self.model_save_dir, "best_by_reward"))
            if train_vec_norm is not None:
                train_vec_norm.save(os.path.join(
                    self.model_save_dir, "best_by_reward_vecnormalize.pkl"))

        return True


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────

def main():
    # ── Validate paths ──
    for p in [REACH_MODEL_PATH, REACH_MEAN_PATH, REACH_VAR_PATH]:
        if not os.path.exists(p):
            print(f"[ERROR] Missing: {p}")
            sys.exit(1)

    env_name = "h1hand-reach-v0"
    model_save_path = f"models/{env_name}/{ARGS.exp_name}"
    os.makedirs(model_save_path, exist_ok=True)

    # ── Compute obs/action dimension mappings ──
    # Create a temporary env to inspect the model structure
    temp_env = gym.make("h1hand-reach-v0")
    obs_mapping = get_obs_mapping(temp_env)
    act_mapping = get_action_mapping(temp_env)
    temp_env.close()
    print(f"Obs mapping:  55D → {len(set(obs_mapping))} unique indices in 155D")
    print(f"Act mapping:  19D → {len(act_mapping)} indices in 61D")

    # ── Training envs ──
    env = SubprocVecEnv([make_env(i) for i in range(ARGS.num_envs)])
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0)

    # ── Eval env ──
    EVAL_SEED_OFFSET = 1000
    eval_env = DummyVecEnv([make_env(EVAL_SEED_OFFSET)])
    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False,
                            clip_obs=10.0, training=False)

    # ── Create model (SB3 auto-detects 155D obs, 61D act from env) ──
    algo_cls = {"ppo": PPO, "grpo": GRPO}[ARGS.algo]

    print(f"Algorithm     : {ARGS.algo}")
    print(f"Reach model   : {REACH_MODEL_PATH}")
    print(f"Obs space     : {env.observation_space}")
    print(f"Action space  : {env.action_space}")

    model = algo_cls(
        "MlpPolicy",
        env,
        verbose=1,
        tensorboard_log=f"runs/{env_name}_{ARGS.exp_name}",
        learning_rate=ARGS.learning_rate,
        n_steps=2048 // ARGS.num_envs,
        batch_size=256,
        policy_kwargs=dict(
            net_arch=dict(pi=[256, 256], vf=[256, 256]),
        ),
        seed=ARGS.seed,
        device="cuda",
    )

    # ── Load pretrained reach weights (with dimension expansion) ──
    print("\nLoading pretrained reach weights (expanding 55→155 obs, 19→61 act)...")
    load_reach_weights_expanded(
        model, REACH_MODEL_PATH, REACH_MEAN_PATH, REACH_VAR_PATH,
        obs_mapping, act_mapping,
    )

    # Sync eval VecNormalize
    train_vec_norm = model.get_vec_normalize_env()
    eval_env.obs_rms = train_vec_norm.obs_rms
    eval_env.ret_rms = RunningMeanStd(shape=())
    eval_env.training = False

    # ── Callbacks ──
    eval_callback = PeriodicEvalCallback(
        eval_env=eval_env,
        model_save_dir=model_save_path,
        eval_freq=ARGS.eval_freq // ARGS.num_envs,
        n_eval_episodes=ARGS.n_eval_episodes,
    )

    try:
        import wandb
        run = wandb.init(
            entity=ARGS.wandb_entity,
            project="humanoid-bench",
            name=f"{ARGS.exp_name}",
            tags=[f"algo_{ARGS.algo}", f"seed_{ARGS.seed}",
                  "reach_sparse", "finetune"],
            sync_tensorboard=True,
            monitor_gym=True,
            save_code=False,
            config=vars(ARGS),
        )
    except Exception:
        run = None
        print("[WARN] wandb not available; skipping.")

    # ── Train ──
    print(f"\nStarting fine-tuning ({ARGS.algo.upper()}) on reach task "
          f"with sparse success reward...")
    try:
        model.learn(
            total_timesteps=ARGS.max_steps,
            log_interval=1,
            reset_num_timesteps=True,
            callback=[
                eval_callback,
                EpisodeLogCallback(),
            ],
        )
    except KeyboardInterrupt:
        print("\nUser interrupted. Saving final state...")
    finally:
        model.save(os.path.join(model_save_path, "latest_model"))
        env.save(os.path.join(model_save_path, "latest_vecnormalize.pkl"))
        env.close()

    print(f"Training finished. Models saved to: {model_save_path}")
    eval_env.close()

    # ── Quick final evaluation (100 episodes) ──
    print("\n" + "=" * 60)
    print("Final Evaluation")
    print("=" * 60)

    FINAL_EVAL_EPS = ARGS.n_eval_episodes * 5  # 100 episodes

    for tag in ["best_by_success", "best_by_reward", "latest_model"]:
        model_file = os.path.join(model_save_path, f"{tag}.zip")
        vecnorm_file = os.path.join(model_save_path, f"{tag}_vecnormalize.pkl")
        if not os.path.exists(model_file):
            print(f"  [SKIP] {tag} — model file not found")
            continue

        print(f"\n[{tag}]")

        eval_env2 = DummyVecEnv([make_env(EVAL_SEED_OFFSET)])

        model2 = algo_cls.load(model_file, env=eval_env2)

        # Load VecNormalize onto the raw DummyVecEnv (single layer).
        # Wrapping a fresh VecNormalize first and then loading on top of it
        # double-normalizes: the inner fresh layer clips raw obs to ±10
        # BEFORE the real stats are applied, corrupting high-magnitude
        # observations (e.g. joint velocities).
        if os.path.exists(vecnorm_file):
            eval_env2 = VecNormalize.load(vecnorm_file, eval_env2)
            eval_env2.training = False
            eval_env2.norm_reward = False
        else:
            eval_env2 = VecNormalize(eval_env2, norm_obs=True, norm_reward=False,
                                     clip_obs=10.0, training=False)

        success_count = 0
        all_returns = []
        all_lengths = []
        all_min_dists = []

        for ep in range(FINAL_EVAL_EPS):
            eval_env2.venv.envs[0].reset(seed=EVAL_SEED_OFFSET + ep)
            obs = eval_env2.reset()
            done = False
            ep_rew = 0.0
            ep_len = 0
            ep_min_dist = float("inf")
            ep_success = False

            while not done:
                action, _ = model2.predict(obs, deterministic=True)
                obs, rewards, dones, infos = eval_env2.step(action)
                ep_rew += float(rewards[0])
                ep_len += 1
                if infos[0].get("success", 0) == 1:
                    ep_success = True
                d = infos[0].get("min_hand_dist", float("inf"))
                if d < ep_min_dist:
                    ep_min_dist = d
                done = bool(dones[0])

            if ep_success:
                success_count += 1
            all_returns.append(ep_rew)
            all_lengths.append(ep_len)
            all_min_dists.append(ep_min_dist)

        eval_env2.close()

        n_total = len(all_returns)
        print(f"  Episodes       : {n_total}")
        print(f"  Success rate   : {success_count / n_total:.3f} "
              f"({success_count}/{n_total})")
        print(f"  Mean return    : {np.mean(all_returns):.1f} ± {np.std(all_returns):.1f}")
        print(f"  Mean length    : {np.mean(all_lengths):.1f} ± {np.std(all_lengths):.1f}")
        print(f"  Mean min dist  : {np.mean(all_min_dists):.4f} ± {np.std(all_min_dists):.4f}")

    if run is not None:
        run.finish()


if __name__ == "__main__":
    main()
