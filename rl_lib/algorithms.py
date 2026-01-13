"""Algorithm implementations for RL training."""

import time
from typing import Any, Dict, Tuple, Type

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812
from torch.amp import autocast

from rl_lib.base import BaseTrainer
from rl_lib.buffers import ReplayBuffer, RolloutBuffer
from rl_lib.networks import ActorNet, CriticNet, IQNQNet, PPOActorNet, QNet, ValueNet

# Algorithm registry
ALGORITHMS: Dict[str, Type[BaseTrainer]] = {}


def register_algorithm(name: str):
    """Decorator to register an algorithm in the registry.

    Args:
        name: Algorithm name
    """
    def decorator(cls: Type[BaseTrainer]):
        ALGORITHMS[name.lower()] = cls
        return cls
    return decorator


def get_algorithm(name: str) -> Type[BaseTrainer]:
    """Get algorithm class by name.

    Args:
        name: Algorithm name (case-insensitive)

    Returns:
        Algorithm class

    Raises:
        ValueError: If algorithm not found
    """
    name_lower = name.lower()
    if name_lower not in ALGORITHMS:
        available = ", ".join(ALGORITHMS.keys())
        raise ValueError(f"Unknown algorithm: {name}. Available: {available}")
    return ALGORITHMS[name_lower]


@register_algorithm("dqn")
class DQNTrainer(BaseTrainer):
    """DQN trainer with soft Double Q-learning and entropy regularization."""

    def __init__(self, config: Dict[str, Any], experiment_name: str = None):
        super().__init__(config, experiment_name)

        # DQN-specific hyperparameters
        self.alpha_start = float(self.training_cfg.get("alpha_start", 0.05))
        self.alpha_final = float(self.training_cfg.get("alpha_final", 0.05))
        self.alpha_decay_steps = int(self.training_cfg.get("alpha_decay_steps", 0))
        self.target_clip = self.training_cfg.get("target_clip", 100)

    def setup_networks(self, obs_shape, n_actions):
        """Setup Q-networks."""
        net_cfg = self.config.get("network", {})
        conv_channels = tuple(net_cfg.get("conv_channels", [32, 64, 64]))
        fc_hidden = int(net_cfg.get("fc_hidden", 512))

        self.q = QNet(n_actions, in_ch=obs_shape[0], conv_channels=conv_channels, fc_hidden=fc_hidden).to(self.device)
        self.q_tgt = QNet(n_actions, in_ch=obs_shape[0], conv_channels=conv_channels, fc_hidden=fc_hidden).to(self.device)
        self.q_tgt.load_state_dict(self.q.state_dict())
        self.q_tgt.eval()

    def _get_alpha(self, step: int) -> float:
        """Get alpha (entropy temperature) for current step.

        Args:
            step: Current training step

        Returns:
            Alpha value
        """
        return self._schedule_helper(step, self.alpha_start, self.alpha_final, self.alpha_decay_steps)

    def train_step(self, batch) -> Tuple[float, float, Dict[str, float]]:
        """Perform one DQN training step.

        Args:
            batch: Tuple of (obs, act, rew, nxt, done)

        Returns:
            Tuple of (loss, grad_norm, metrics_dict)
        """
        obs, act, rew, nxt, done = batch

        # Transfers (use pinned memory only for CUDA)
        obs_cpu = torch.from_numpy(obs)
        nxt_cpu = torch.from_numpy(nxt)
        act_cpu = torch.from_numpy(act)
        rew_cpu = torch.from_numpy(rew)
        done_cpu = torch.from_numpy(done)

        if self.device == "cuda":
            obs_cpu = obs_cpu.pin_memory()
            nxt_cpu = nxt_cpu.pin_memory()
            act_cpu = act_cpu.pin_memory()
            rew_cpu = rew_cpu.pin_memory()
            done_cpu = done_cpu.pin_memory()

        non_blocking = (self.device == "cuda")
        obs = obs_cpu.to(self.device, non_blocking=non_blocking)
        nxt = nxt_cpu.to(self.device, non_blocking=non_blocking)
        act = act_cpu.to(self.device, dtype=torch.long, non_blocking=non_blocking)
        rew = rew_cpu.to(self.device, dtype=torch.float32, non_blocking=non_blocking)
        done = done_cpu.to(self.device, dtype=torch.float32, non_blocking=non_blocking)

        self.opt.zero_grad(set_to_none=True)

        device_type = self.device if self.device in ("cuda", "cpu") else "cuda"
        with autocast(device_type=device_type, enabled=self.use_amp):
            q_sa = self.q(obs).gather(1, act[:, None]).squeeze(1)

            with torch.no_grad():
                q_next = self.q(nxt)
                q_next_t = self.q_tgt(nxt)

                # Boltzmann policy for Double Q-learning
                alpha = max(self._get_alpha(self._get_training_step()), 1e-6)
                logits = q_next / alpha
                logits -= logits.max(1, keepdim=True).values
                pi = torch.softmax(logits, dim=1)

                entropy = -(pi * (pi + 1e-8).log()).sum(1)
                v_next = (pi * q_next_t).sum(1) - alpha * entropy

                target = rew + self.gamma * (1 - done) * v_next
                if self.target_clip is not None:
                    target = target.clamp(-float(self.target_clip), float(self.target_clip))

            loss = F.smooth_l1_loss(q_sa, target)

            # Monitoring metrics
            with torch.no_grad():
                td = q_sa - target
                metrics = {
                    "td_error_mae": torch.mean(torch.abs(td)).float().item(),
                    "td_error_mse": torch.mean(td * td).float().item(),
                    "q_sa_max": torch.max(q_sa).float().item(),
                    "q_sa_min": torch.min(q_sa).float().item(),
                    "target_max": torch.max(target).float().item(),
                    "target_min": torch.min(target).float().item(),
                    "entropy_mean": torch.mean(entropy).float().item(),
                    "entropy_min": torch.min(entropy).float().item(),
                    "entropy_max": torch.max(entropy).float().item(),
                    "alpha": alpha,
                }

        if self.scaler is not None:
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.opt)
            grad_norm = nn.utils.clip_grad_norm_(self.q.parameters(), self.grad_clip)
            self.scaler.step(self.opt)
            self.scaler.update()
        else:
            loss.backward()
            grad_norm = nn.utils.clip_grad_norm_(self.q.parameters(), self.grad_clip)
            self.opt.step()

        grad_norm = grad_norm.item() if torch.is_tensor(grad_norm) else float(grad_norm)
        return loss.item(), grad_norm, metrics

    def select_actions(self, obs):
        """Select actions using Boltzmann policy.

        Args:
            obs: Observations [num_envs, C, H, W]

        Returns:
            Actions as numpy array
        """
        alpha = max(self._get_alpha(self._get_training_step()), 1e-6)

        obs_cpu = torch.from_numpy(obs)
        if self.device == "cuda":
            obs_cpu = obs_cpu.pin_memory()
        obs = obs_cpu.to(self.device, non_blocking=(self.device == "cuda"))

        with torch.no_grad():
            logits = self.q(obs) / alpha
            logits -= logits.max(1, keepdim=True).values
            probs = torch.softmax(logits, dim=1)
            actions = torch.multinomial(probs, 1).squeeze(1)

        return actions.cpu().numpy()

    def _get_training_step(self) -> int:
        """Get current training step. Override if tracking step differently."""
        # This is a placeholder - the actual step is tracked in run()
        # For now, return a default value
        return 0


@register_algorithm("iqn")
class IQNTrainer(DQNTrainer):
    """IQN (Implicit Quantile Networks) trainer.

    Extends DQN with distributional RL using quantile regression.
    """

    def __init__(self, config: Dict[str, Any], experiment_name: str = None):
        super().__init__(config, experiment_name)

        # IQN-specific hyperparameters
        iqn_cfg = config.get("iqn", {})
        self.num_quantiles = int(iqn_cfg.get("num_quantiles", 32))
        self.quantile_embed_dim = int(iqn_cfg.get("embed_dim", 64))
        self.kappa = float(iqn_cfg.get("kappa", 1.0))
        self.eval_quantiles = int(iqn_cfg.get("eval_quantiles", self.num_quantiles))

    def setup_networks(self, obs_shape, n_actions):
        """Setup IQN networks."""
        net_cfg = self.config.get("network", {})
        conv_channels = tuple(net_cfg.get("conv_channels", [32, 64, 64]))
        fc_hidden = int(net_cfg.get("fc_hidden", 512))

        self.q = IQNQNet(
            n_actions,
            in_ch=obs_shape[0],
            conv_channels=conv_channels,
            fc_hidden=fc_hidden,
            quantile_embed_dim=self.quantile_embed_dim,
        ).to(self.device)

        self.q_tgt = IQNQNet(
            n_actions,
            in_ch=obs_shape[0],
            conv_channels=conv_channels,
            fc_hidden=fc_hidden,
            quantile_embed_dim=self.quantile_embed_dim,
        ).to(self.device)

        self.q_tgt.load_state_dict(self.q.state_dict())
        self.q_tgt.eval()

    def train_step(self, batch) -> Tuple[float, float, Dict[str, float]]:
        """Perform one IQN training step with quantile regression.

        Args:
            batch: Tuple of (obs, act, rew, nxt, done)

        Returns:
            Tuple of (loss, grad_norm, metrics_dict)
        """
        obs, act, rew, nxt, done = batch

        # Transfers (use pinned memory only for CUDA)
        obs_cpu = torch.from_numpy(obs)
        nxt_cpu = torch.from_numpy(nxt)
        act_cpu = torch.from_numpy(act)
        rew_cpu = torch.from_numpy(rew)
        done_cpu = torch.from_numpy(done)

        if self.device == "cuda":
            obs_cpu = obs_cpu.pin_memory()
            nxt_cpu = nxt_cpu.pin_memory()
            act_cpu = act_cpu.pin_memory()
            rew_cpu = rew_cpu.pin_memory()
            done_cpu = done_cpu.pin_memory()

        non_blocking = (self.device == "cuda")
        obs = obs_cpu.to(self.device, non_blocking=non_blocking)
        nxt = nxt_cpu.to(self.device, non_blocking=non_blocking)
        act = act_cpu.to(self.device, dtype=torch.long, non_blocking=non_blocking)
        rew = rew_cpu.to(self.device, dtype=torch.float32, non_blocking=non_blocking)
        done = done_cpu.to(self.device, dtype=torch.float32, non_blocking=non_blocking)

        self.opt.zero_grad(set_to_none=True)

        b = obs.shape[0]

        device_type = self.device if self.device in ("cuda", "cpu") else "cuda"
        with autocast(device_type=device_type, enabled=self.use_amp):
            # Sample taus for prediction and target
            taus_pred = torch.rand(b, self.num_quantiles, device=obs.device)
            taus_tgt = torch.rand(b, self.num_quantiles, device=obs.device)

            # Predicted quantiles for chosen actions
            q_pred_all = self.q(obs, taus_pred)  # [b, N, A]
            q_pred_sa = q_pred_all.gather(2, act[:, None, None].expand(b, self.num_quantiles, 1)).squeeze(2)  # [b, N]

            # Next-state greedy actions via expected Q (Double IQN)
            q_next_exp = self.q.expected_q(nxt, num_quantiles=self.num_quantiles)  # [b, A]
            a_star = torch.argmax(q_next_exp, dim=1)  # [b]

            # Target quantiles from target network
            q_tgt_all = self.q_tgt(nxt, taus_tgt)  # [b, N, A]
            q_tgt_star = q_tgt_all.gather(2, a_star[:, None, None].expand(b, self.num_quantiles, 1)).squeeze(2)  # [b, N]
            target = rew[:, None] + self.gamma * (1.0 - done)[:, None] * q_tgt_star  # [b, N]

            # Quantile regression Huber loss
            u = target[:, None, :] - q_pred_sa[:, :, None]  # [b, N, N]
            abs_u = torch.abs(u)
            huber = torch.where(abs_u <= self.kappa, 0.5 * u * u, self.kappa * (abs_u - 0.5 * self.kappa))

            # Weighting by tau - I[u<0]
            tau = taus_pred[:, :, None]  # [b, N, 1]
            weight = torch.abs(tau - (u.detach() < 0.0).float())  # [b, N, N]
            loss = (weight * huber).mean()

            metrics = {
                "iqn_loss": loss.detach().item(),
            }

        if self.scaler is not None:
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.opt)
            grad_norm = nn.utils.clip_grad_norm_(self.q.parameters(), self.grad_clip)
            self.scaler.step(self.opt)
            self.scaler.update()
        else:
            loss.backward()
            grad_norm = nn.utils.clip_grad_norm_(self.q.parameters(), self.grad_clip)
            self.opt.step()

        grad_norm = grad_norm.item() if torch.is_tensor(grad_norm) else float(grad_norm)
        return loss.item(), grad_norm, metrics

    def select_actions(self, obs):
        """Select actions using expected Q-values over quantiles.

        Args:
            obs: Observations [num_envs, C, H, W]

        Returns:
            Actions as numpy array
        """
        obs_cpu = torch.from_numpy(obs)
        if self.device == "cuda":
            obs_cpu = obs_cpu.pin_memory()
        obs = obs_cpu.to(self.device, non_blocking=(self.device == "cuda"))

        with torch.no_grad():
            q_vals = self.q.expected_q(obs, num_quantiles=self.eval_quantiles)
            actions = q_vals.argmax(dim=1)

        return actions.cpu().numpy()


# =========================
# PPO (Proximal Policy Optimization)
# =========================
@register_algorithm("ppo")
class PPOTrainer(BaseTrainer):
    """PPO trainer for continuous control (MuJoCo)."""

    def __init__(self, config: Dict[str, Any], experiment_name: str = None):
        super().__init__(config, experiment_name)

        # PPO-specific hyperparameters
        ppo_cfg = config.get("ppo", {})
        self.gae_lambda = float(ppo_cfg.get("gae_lambda", 0.95))
        self.clip_range = float(ppo_cfg.get("clip_range", 0.2))
        self.entropy_coef = float(ppo_cfg.get("entropy_coef", 0.0))
        self.vf_coef = float(ppo_cfg.get("vf_coef", 0.5))
        self.max_grad_norm = float(ppo_cfg.get("max_grad_norm", 0.5))
        self.rollout_steps = int(ppo_cfg.get("rollout_steps", 2048))
        self.num_epochs = int(ppo_cfg.get("num_epochs", 10))
        self.minibatch_size = int(ppo_cfg.get("minibatch_size", 64))

        # Use smaller buffer size for PPO (on-policy)
        self.buffer_size = self.rollout_steps

    def setup_networks(self, obs_shape, n_actions):
        """Setup PPO networks (actor and value)."""
        net_cfg = self.config.get("network", {})
        hidden_dim = int(net_cfg.get("hidden_dim", 64))
        hidden_layers = int(net_cfg.get("hidden_layers", 2))

        # For continuous actions, n_actions is the action dimension
        self.actor = PPOActorNet(obs_shape[0], n_actions, hidden_dim, hidden_layers).to(self.device)
        self.value = ValueNet(obs_shape[0], hidden_dim, hidden_layers).to(self.device)

        # Create aliases for compatibility with BaseTrainer
        self.q = self.actor  # For checkpointing
        self.q_tgt = self.value  # For checkpointing

    def setup_envs(self):
        """Override to use fewer envs for PPO."""
        from gymnasium.vector import SyncVectorEnv

        from rl_lib.envs import is_mujoco_env

        if is_mujoco_env(self.env_name):
            # Use sync vector env for MuJoCo (faster for small num_envs)
            envs = SyncVectorEnv([
                self._make_env(i) for i in range(self.num_envs)
            ])
        else:
            envs = super().setup_envs()
        return envs

    def _make_env(self, rank):
        """Helper to create a single environment."""
        from rl_lib.envs import make_env
        max_episode_steps = self.env_cfg.get("max_episode_steps")
        return make_env(self.env_name, self.seed_base + rank, 0, 0, 0, max_episode_steps)()

    def _compute_gae(self, rewards, values, dones, next_value):
        """Compute Generalized Advantage Estimation (GAE).

        Args:
            rewards: [rollout_steps, num_envs]
            values: [rollout_steps, num_envs]
            dones: [rollout_steps, num_envs]
            next_value: [num_envs]

        Returns:
            advantages, returns
        """
        advantages = np.zeros_like(rewards)
        last_advantage = 0

        # Reverse pass for GAE
        for t in reversed(range(self.rollout_steps)):
            if t == self.rollout_steps - 1:
                next_value_t = next_value
                next_non_terminal = 1.0 - dones[t]
            else:
                next_value_t = values[t + 1]
                next_non_terminal = 1.0 - dones[t]

            delta = rewards[t] + self.gamma * next_value_t * next_non_terminal - values[t]
            advantages[t] = last_advantage = delta + self.gamma * self.gae_lambda * next_non_terminal * last_advantage

        returns = advantages + values
        return advantages, returns

    def _collect_rollouts(self, envs, obs, rollout_buffer, step):
        """Collect rollouts using current policy.

        Returns:
            next_obs, episode_returns
        """
        episode_returns = []

        for _ in range(self.rollout_steps):
            with torch.no_grad():
                obs_tensor = torch.from_numpy(obs).to(self.device).float()
                policy_out = self.actor(obs_tensor)
                actions = policy_out["action"].cpu().numpy()
                log_probs = policy_out["log_prob"].squeeze(-1).cpu().numpy()
                values = self.value(obs_tensor).squeeze(-1).cpu().numpy()

            # Step environment
            next_obs, rewards, terms, truncs, _ = envs.step(actions)
            dones = np.logical_or(terms, truncs).astype(np.float32)

            # Store in buffer
            rollout_buffer.add(obs, actions, rewards, values, log_probs, dones)

            obs = next_obs

            # Track episode returns
            for i, done in enumerate(dones):
                if done:
                    # Get episode return (simplified - we should track this properly)
                    episode_returns.append(rewards[i])

        # Compute next value for GAE
        with torch.no_grad():
            obs_tensor = torch.from_numpy(obs).to(self.device).float()
            next_value = self.value(obs_tensor).squeeze(-1).cpu().numpy()

        return obs, next_value, episode_returns

    def _update_policy(self, rollout_buffer, next_value):
        """Update policy using PPO loss."""
        data = rollout_buffer.get()

        obs = torch.from_numpy(data["obs"]).to(self.device).float()
        actions = torch.from_numpy(data["actions"]).to(self.device).float()
        old_log_probs = torch.from_numpy(data["log_probs"]).to(self.device).float()
        old_values = torch.from_numpy(data["values"]).to(self.device).float()
        rewards = data["rewards"]
        dones = data["dones"]

        # Reshape for GAE computation
        rewards = rewards.reshape(self.rollout_steps, self.num_envs)
        dones = dones.reshape(self.rollout_steps, self.num_envs)
        old_values = old_values.reshape(self.rollout_steps, self.num_envs)
        next_value = next_value.reshape(self.num_envs)

        # Compute GAE
        advantages, returns = self._compute_gae(rewards, old_values, dones, next_value)

        # Flatten
        advantages = advantages.flatten()
        returns = returns.flatten()

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        total_metrics = {}

        # Multiple epochs over the same data
        for epoch in range(self.num_epochs):
            # Generate random indices
            indices = np.arange(len(obs))
            np.random.shuffle(indices)

            # Minibatch updates
            for start in range(0, len(obs), self.minibatch_size):
                end = start + self.minibatch_size
                mb_indices = indices[start:end]

                mb_obs = obs[mb_indices]
                mb_actions = actions[mb_indices]
                mb_old_log_probs = old_log_probs.flatten()[mb_indices]
                mb_advantages = advantages[mb_indices]
                mb_returns = returns[mb_indices]

                # Evaluate current policy
                policy_out = self.actor(mb_obs, mb_actions)
                new_log_probs = policy_out["log_prob"].squeeze(-1)
                entropy = policy_out["entropy"].mean()

                new_values = self.value(mb_obs).squeeze(-1)

                # Compute ratio
                ratio = torch.exp(new_log_probs - mb_old_log_probs)

                # PPO clip loss
                surr1 = ratio * mb_advantages
                surr2 = torch.clamp(ratio, 1.0 - self.clip_range, 1.0 + self.clip_range) * mb_advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value loss
                value_loss = F.mse_loss(new_values, mb_returns)

                # Entropy bonus
                entropy_loss = -self.entropy_coef * entropy

                # Total loss
                loss = policy_loss + self.vf_coef * value_loss + entropy_loss

                # Optimize
                self.opt.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(
                    list(self.actor.parameters()) + list(self.value.parameters()),
                    self.max_grad_norm
                )
                self.opt.step()

                # Metrics (only log first epoch)
                if epoch == 0:
                    total_metrics["policy_loss"] = policy_loss.item()
                    total_metrics["value_loss"] = value_loss.item()
                    total_metrics["entropy"] = entropy.item()
                    total_metrics["clip_frac"] = ((ratio - 1.0).abs() > self.clip_range).float().mean().item()

        return total_metrics

    def train_step(self, batch):
        """PPO doesn't use train_step in the same way."""
        raise NotImplementedError("PPO uses custom training loop")

    def select_actions(self, obs):
        """Select actions using current policy."""
        obs_tensor = torch.from_numpy(obs).to(self.device).float()
        with torch.no_grad():
            policy_out = self.actor(obs_tensor)
            actions = policy_out["action"].cpu().numpy()
        return actions

    def run(self):
        """Custom run loop for PPO."""
        # Setup
        envs = self.setup_envs()
        obs, _ = envs.reset()

        try:
            action_dim = int(envs.single_action_space.shape[0])
        except Exception:
            raise RuntimeError("Expected Box action space.")

        # Setup networks
        self.setup_networks(obs.shape[1:], action_dim)

        # Setup optimizer
        self.opt = torch.optim.Adam(
            list(self.actor.parameters()) + list(self.value.parameters()),
            lr=self.lr_start
        )

        # Setup rollout buffer
        rollout_buffer = RolloutBuffer(
            self.num_envs, self.rollout_steps, obs.shape[1:], action_dim
        )

        # Setup logging
        from torch.utils.tensorboard.writer import SummaryWriter

        from rl_lib.config import save_config

        writer = SummaryWriter(self.log_dir)
        save_config(self.config, self.save_dir / "config.yaml")

        step = 0
        episode_rewards = np.zeros(self.num_envs)
        start_time = time.time()
        last_log_step = 0
        last_log_time = start_time

        print(f"Starting PPO training on {self.env_name}")
        print(f"Observation shape: {obs.shape[1:]}, Action dim: {action_dim}")

        while step < self.total_steps:
            # Update learning rate
            lr_now = self._update_learning_rate(step)

            # Collect rollouts
            obs, next_value, rollout_returns = self._collect_rollouts(envs, obs, rollout_buffer, step)

            # Update policy
            metrics = self._update_policy(rollout_buffer, next_value)

            # Reset buffer
            rollout_buffer.reset()

            # Update step count
            steps_collected = self.rollout_steps * self.num_envs
            step += steps_collected

            # Track episode stats
            episode_rewards += rollout_returns  # Simplified

            # Logging
            if step - last_log_step >= self.log_interval:
                elapsed = time.time() - last_log_time
                sps = (step - last_log_step) / max(elapsed, 1e-6)

                print(f"Step {step:,} | SPS {sps:.1f}")

                for key, value in metrics.items():
                    writer.add_scalar(f"train/{key}", value, step)
                writer.add_scalar("train/lr", lr_now, step)
                writer.add_scalar("perf/steps_per_sec", sps, step)

                last_log_step = step
                last_log_time = time.time()

            # Checkpoint
            if step % self.save_interval == 0:
                self._save_checkpoint(step)

        # Save final model
        torch.save({
            "actor_state_dict": self.actor.state_dict(),
            "value_state_dict": self.value.state_dict(),
            "config": self.config,
        }, self.save_dir / "final_ppo_model.pt")
        print("Training complete.")


# =========================
# SAC (Soft Actor-Critic)
# =========================
@register_algorithm("sac")
class SACTrainer(BaseTrainer):
    """SAC trainer for continuous control (MuJoCo)."""

    def __init__(self, config: Dict[str, Any], experiment_name: str = None):
        super().__init__(config, experiment_name)

        # SAC-specific hyperparameters
        sac_cfg = config.get("sac", {})
        self.tau = float(sac_cfg.get("tau", 0.005))
        self.target_update_freq = int(sac_cfg.get("target_update_freq", 1))
        self.target_entropy = float(sac_cfg.get("target_entropy", -float("inf")))
        self.autotune = bool(sac_cfg.get("autotune", True))

        # For continuous action spaces
        try:
            from rl_lib.envs import is_mujoco_env
            self.is_mujoco = is_mujoco_env(self.env_name)
        except Exception:
            self.is_mujoco = False

    def setup_networks(self, obs_shape, n_actions):
        """Setup SAC networks (actor and double critic)."""
        net_cfg = self.config.get("network", {})
        sac_cfg = self.config.get("sac", {})
        hidden_dim = int(net_cfg.get("hidden_dim", 256))
        hidden_layers = int(net_cfg.get("hidden_layers", 2))

        # Actor network
        self.actor = ActorNet(obs_shape[0], n_actions, hidden_dim, hidden_layers).to(self.device)

        # Critic networks (double Q)
        self.critic = CriticNet(obs_shape[0], n_actions, hidden_dim, hidden_layers).to(self.device)
        self.critic_tgt = CriticNet(obs_shape[0], n_actions, hidden_dim, hidden_layers).to(self.device)
        self.critic_tgt.load_state_dict(self.critic.state_dict())
        self.critic_tgt.eval()

        # Create aliases for compatibility
        self.q = self.actor
        self.q_tgt = self.critic_tgt

        # Automatic entropy tuning
        if self.autotune and self.target_entropy == -float("inf"):
            self.target_entropy = -n_actions
        if self.autotune:
            self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
            self.alpha_opt = torch.optim.Adam([self.log_alpha], lr=self.lr_start)
        else:
            self.log_alpha = None
            self.alpha = sac_cfg.get("alpha", 0.2)

    def setup_envs(self):
        """Override to use fewer envs for SAC."""
        from gymnasium.vector import SyncVectorEnv

        from rl_lib.envs import is_mujoco_env

        if is_mujoco_env(self.env_name):
            # Use sync vector env for MuJoCo
            envs = SyncVectorEnv([
                self._make_env(i) for i in range(self.num_envs)
            ])
        else:
            envs = super().setup_envs()
        return envs

    def _make_env(self, rank):
        """Helper to create a single environment."""
        from rl_lib.envs import make_env
        max_episode_steps = self.env_cfg.get("max_episode_steps")
        return make_env(self.env_name, self.seed_base + rank, 0, 0, 0, max_episode_steps)()

    def train_step(self, batch):
        """Perform one SAC training step."""
        obs, act, rew, nxt, done = batch

        # Transfer to device
        obs = torch.from_numpy(obs).to(self.device).float()
        nxt = torch.from_numpy(nxt).to(self.device).float()
        act = torch.from_numpy(act).to(self.device).float()
        rew = torch.from_numpy(rew).to(self.device).float().unsqueeze(-1)
        done = torch.from_numpy(done).to(self.device).float().unsqueeze(-1)

        # Get current alpha
        if self.autotune:
            alpha = torch.exp(self.log_alpha).detach()
        else:
            alpha = self.alpha

        self.opt.zero_grad()

        # Critic loss
        with torch.no_grad():
            # Next actions and log probs
            next_actions, next_log_probs = self.actor(nxt)
            next_q1, next_q2 = self.critic_tgt(nxt, next_actions)
            next_q = torch.min(next_q1, next_q2) - alpha * next_log_probs
            target = rew + self.gamma * (1 - done) * next_q

        q1, q2 = self.critic(obs, act)
        critic_loss = F.mse_loss(q1, target) + F.mse_loss(q2, target)

        # Actor loss
        actions, log_probs = self.actor(obs)
        q1_new, q2_new = self.critic(obs, actions)
        q_new = torch.min(q1_new, q2_new)

        actor_loss = (alpha * log_probs - q_new).mean()

        # Alpha loss (autotune)
        alpha_loss = torch.tensor(0.0)
        if self.autotune:
            alpha_loss = -(self.log_alpha * (log_probs + self.target_entropy).detach()).mean()

        # Total loss
        loss = critic_loss + actor_loss + alpha_loss

        # Optimize critic and actor
        loss.backward()
        nn.utils.clip_grad_norm_(
            list(self.actor.parameters()) + list(self.critic.parameters()),
            self.grad_clip
        )
        self.opt.step()

        # Update alpha (autotune)
        if self.autotune:
            self.alpha_opt.zero_grad()
            alpha_loss.backward()
            self.alpha_opt.step()

        # Soft update target network
        with torch.no_grad():
            for p_tgt, p in zip(self.critic_tgt.parameters(), self.critic.parameters()):
                p_tgt.data.mul_(1.0 - self.tau).add_(p.data, alpha=self.tau)

        # Metrics
        metrics = {
            "critic_loss": critic_loss.item(),
            "actor_loss": actor_loss.item(),
            "q_value": q_new.mean().item(),
            "log_prob": log_probs.mean().item(),
        }
        if self.autotune:
            metrics["alpha"] = alpha.item()
            metrics["alpha_loss"] = alpha_loss.item()

        return loss.item(), 0.0, metrics

    def select_actions(self, obs):
        """Select actions with optional exploration noise."""
        obs_tensor = torch.from_numpy(obs).to(self.device).float()

        with torch.no_grad():
            # Small amount of noise during training
            if self.training:
                actions, _ = self.actor(obs_tensor)
            else:
                actions, _ = self.actor(obs_tensor, deterministic=True)

            actions = actions.cpu().numpy()

        return actions

    def run(self):
        """Custom run loop for SAC."""
        import time

        from torch.utils.tensorboard.writer import SummaryWriter

        from rl_lib.config import save_config

        # Setup
        envs = self.setup_envs()
        obs, _ = envs.reset()

        try:
            action_dim = int(envs.single_action_space.shape[0])
        except Exception:
            raise RuntimeError("Expected Box action space.")

        # Setup networks
        self.setup_networks(obs.shape[1:], action_dim)

        # Setup optimizer (actor + critic)
        self.opt = torch.optim.Adam(
            list(self.actor.parameters()) + list(self.critic.parameters()),
            lr=self.lr_start
        )

        # Setup replay buffer (float for MuJoCo)
        rb = ReplayBuffer(obs.shape[1:], self.buffer_size)
        # Override obs storage for float32
        if self.is_mujoco:
            rb.obs = np.empty((rb.size, *obs.shape[1:]), np.float32)
            rb.next = np.empty_like(rb.obs)

        # Setup logging
        writer = SummaryWriter(self.log_dir)
        save_config(self.config, self.save_dir / "config.yaml")

        step = 0
        completed_episodes = 0
        episode_rewards = np.zeros(self.num_envs)
        episode_lengths = np.zeros(self.num_envs, dtype=np.int32)
        start_time = time.time()
        last_log_step = 0
        last_log_time = start_time

        self.training = True

        print(f"Starting SAC training on {self.env_name}")
        print(f"Observation shape: {obs.shape[1:]}, Action dim: {action_dim}")

        while step < self.total_steps:
            # Update learning rate
            lr_now = self._update_learning_rate(step)

            # Select and step actions
            actions = self.select_actions(obs)

            nxt, rew, term, trunc, _ = envs.step(actions)
            done = np.logical_or(term, trunc).astype(np.float32)

            # Store transitions
            rb.push_batch(obs, actions, rew, nxt, done)

            # Track episode stats
            episode_rewards += rew
            episode_lengths += 1
            for i, d in enumerate(done):
                if d:
                    writer.add_scalar("episode/reward", episode_rewards[i], completed_episodes)
                    writer.add_scalar("episode/length", int(episode_lengths[i]), completed_episodes)
                    completed_episodes += 1
                    episode_rewards[i] = 0.0
                    episode_lengths[i] = 0

            obs = nxt
            step += self.num_envs

            # Training
            if len(rb) >= self.warmup:
                for _ in range(self.updates_per_step):
                    batch = rb.sample(self.batch_size)
                    loss, grad_norm, metrics = self.train_step(batch)

                    writer.add_scalar("train/loss", loss, step)
                    for key, value in metrics.items():
                        writer.add_scalar(f"train/{key}", value, step)

            # Logging
            if step - last_log_step >= self.log_interval:
                elapsed = time.time() - last_log_time
                sps = (step - last_log_step) / max(elapsed, 1e-6)

                print(f"Step {step:,} | SPS {sps:.1f} | Buffer {len(rb)}")
                writer.add_scalar("train/lr", lr_now, step)
                writer.add_scalar("perf/steps_per_sec", sps, step)

                last_log_step = step
                last_log_time = time.time()

            # Checkpoint
            if step % self.save_interval == 0:
                self._save_checkpoint(step)

        # Save final model
        torch.save({
            "actor_state_dict": self.actor.state_dict(),
            "critic_state_dict": self.critic.state_dict(),
            "config": self.config,
        }, self.save_dir / "final_sac_model.pt")
        print("Training complete.")
