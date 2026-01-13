"""Algorithm implementations for RL training."""

from typing import Any, Dict, Tuple, Type

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast

from rl_lib.base import BaseTrainer
from rl_lib.networks import IQNQNet, QNet

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

        B = obs.shape[0]

        device_type = self.device if self.device in ("cuda", "cpu") else "cuda"
        with autocast(device_type=device_type, enabled=self.use_amp):
            # Sample taus for prediction and target
            taus_pred = torch.rand(B, self.num_quantiles, device=obs.device)
            taus_tgt = torch.rand(B, self.num_quantiles, device=obs.device)

            # Predicted quantiles for chosen actions
            q_pred_all = self.q(obs, taus_pred)  # [B, N, A]
            q_pred_sa = q_pred_all.gather(2, act[:, None, None].expand(B, self.num_quantiles, 1)).squeeze(2)  # [B, N]

            # Next-state greedy actions via expected Q (Double IQN)
            q_next_exp = self.q.expected_q(nxt, num_quantiles=self.num_quantiles)  # [B, A]
            a_star = torch.argmax(q_next_exp, dim=1)  # [B]

            # Target quantiles from target network
            q_tgt_all = self.q_tgt(nxt, taus_tgt)  # [B, N, A]
            q_tgt_star = q_tgt_all.gather(2, a_star[:, None, None].expand(B, self.num_quantiles, 1)).squeeze(2)  # [B, N]
            target = rew[:, None] + self.gamma * (1.0 - done)[:, None] * q_tgt_star  # [B, N]

            # Quantile regression Huber loss
            u = target[:, None, :] - q_pred_sa[:, :, None]  # [B, N, N]
            abs_u = torch.abs(u)
            huber = torch.where(abs_u <= self.kappa, 0.5 * u * u, self.kappa * (abs_u - 0.5 * self.kappa))

            # Weighting by tau - I[u<0]
            tau = taus_pred[:, :, None]  # [B, N, 1]
            weight = torch.abs(tau - (u.detach() < 0.0).float())  # [B, N, N]
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
