"""Neural network architectures for RL."""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# =========================
# MLP Networks for MuJoCo
# =========================
class MLP(nn.Module):
    """Simple MLP network."""

    def __init__(self, input_dim, hidden_dim, output_dim, hidden_layers=2):
        super().__init__()
        layers = []
        in_dim = input_dim
        for _ in range(hidden_layers):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU())
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


# =========================
# Actor Network (Policy)
# =========================
class ActorNet(nn.Module):
    """Actor network for continuous action spaces (SAC-style).

    Outputs mean actions with learnable log_std.
    """

    def __init__(self, obs_dim, action_dim, hidden_dim=256, hidden_layers=2,
                 log_std_min=-20.0, log_std_max=2.0):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        self.backbone = MLP(obs_dim, hidden_dim, hidden_dim, hidden_layers)
        self.mean = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Linear(hidden_dim, action_dim)

    def forward(self, obs, deterministic=False):
        """Forward pass.

        Args:
            obs: Observations [B, obs_dim]
            deterministic: If True, return mean actions

        Returns:
            actions, log_probs (or None if deterministic)
        """
        x = self.backbone(obs)
        mean = self.mean(x)
        log_std = self.log_std(x)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)

        if deterministic:
            return torch.tanh(mean), None

        std = torch.exp(log_std)
        dist = torch.distributions.Normal(mean, std)
        x_t = dist.rsample()
        action = torch.tanh(x_t)

        # Compute log probability with Tanh correction
        log_prob = dist.log_prob(x_t)
        log_prob -= torch.log(1 - action.pow(2) + 1e-7)
        log_prob = log_prob.sum(dim=-1, keepdim=True)

        return action, log_prob

    def get_log_std(self, obs):
        """Get log std for given observations (for entropy computation)."""
        x = self.backbone(obs)
        log_std = self.log_std(x)
        return torch.clamp(log_std, self.log_std_min, self.log_std_max)


# =========================
# PPO Actor Network
# =========================
class PPOActorNet(nn.Module):
    """Actor network for PPO with Tanh policy.

    Uses a fixed log_std for simplicity.
    """

    def __init__(self, obs_dim, action_dim, hidden_dim=64, hidden_layers=2,
                 init_log_std=0.0):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim

        self.backbone = MLP(obs_dim, hidden_dim, hidden_dim, hidden_layers)
        self.mean = nn.Linear(hidden_dim, action_dim)

        # Learnable log_std (shared across all actions)
        self.log_std = nn.Parameter(torch.ones(action_dim) * init_log_std)

    def forward(self, obs, action=None):
        """Forward pass.

        Args:
            obs: Observations [B, obs_dim]
            action: Optional actions to evaluate [B, action_dim]

        Returns:
            dict with mean, log_std, actions (if action provided, includes log_prob, entropy)
        """
        x = self.backbone(obs)
        mean = torch.tanh(self.mean(x))
        std = torch.exp(self.log_std.clamp(-20, 2))

        dist = torch.distributions.Normal(mean, std)

        if action is None:
            action = dist.sample()

        log_prob = dist.log_prob(action).sum(dim=-1, keepdim=True)
        entropy = dist.entropy().sum(dim=-1, keepdim=True)

        return {
            "mean": mean,
            "action": action,
            "log_prob": log_prob,
            "entropy": entropy,
        }


# =========================
# Critic Network (Q-function)
# =========================
class CriticNet(nn.Module):
    """Critic network for continuous action spaces (Q-function).

    Uses double Q-learning (two Q networks).
    """

    def __init__(self, obs_dim, action_dim, hidden_dim=256, hidden_layers=2):
        super().__init__()
        input_dim = obs_dim + action_dim

        self.q1 = MLP(input_dim, hidden_dim, 1, hidden_layers)
        self.q2 = MLP(input_dim, hidden_dim, 1, hidden_layers)

    def forward(self, obs, action):
        """Forward pass.

        Args:
            obs: Observations [B, obs_dim]
            action: Actions [B, action_dim]

        Returns:
            q1, q2 values [B, 1]
        """
        x = torch.cat([obs, action], dim=-1)
        return self.q1(x), self.q2(x)


# =========================
# Value Network (V-function)
# =========================
class ValueNet(nn.Module):
    """Value network for PPO."""

    def __init__(self, obs_dim, hidden_dim=64, hidden_layers=2):
        super().__init__()
        self.net = MLP(obs_dim, hidden_dim, 1, hidden_layers)

    def forward(self, obs):
        """Forward pass.

        Args:
            obs: Observations [B, obs_dim]

        Returns:
            values [B, 1]
        """
        return self.net(obs)


# =========================
# Q Network (Nature CNN)
# =========================
class QNet(nn.Module):
    def __init__(self, n_actions, in_ch=4, conv_channels=(32, 64, 64), fc_hidden=512):
        super().__init__()
        c1, c2, c3 = tuple(conv_channels) if len(conv_channels) == 3 else (32, 64, 64)
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, c1, 8, stride=4),
            nn.ReLU(),
            nn.Conv2d(c1, c2, 4, stride=2),
            nn.ReLU(),
            nn.Conv2d(c2, c3, 3, stride=1),
            nn.ReLU(),
        )
        self.fc = nn.Sequential(
            nn.Linear(7 * 7 * c3, fc_hidden),
            nn.ReLU(),
            nn.Linear(fc_hidden, n_actions),
        )

    def forward(self, x):
        x = x.float() / 255.0
        x = self.conv(x)
        return self.fc(x.flatten(1))


# =========================
# IQN Q Network (Implicit Quantile Networks)
# =========================
class IQNQNet(nn.Module):
    def __init__(self, n_actions, in_ch=4, conv_channels=(32, 64, 64), fc_hidden=512, quantile_embed_dim=64):
        super().__init__()
        c1, c2, c3 = tuple(conv_channels) if len(conv_channels) == 3 else (32, 64, 64)
        self.n_actions = int(n_actions)
        self.quantile_embed_dim = int(quantile_embed_dim)

        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, c1, 8, stride=4),
            nn.ReLU(),
            nn.Conv2d(c1, c2, 4, stride=2),
            nn.ReLU(),
            nn.Conv2d(c2, c3, 3, stride=1),
            nn.ReLU(),
        )
        self.state_fc = nn.Sequential(
            nn.Linear(7 * 7 * c3, fc_hidden),
            nn.ReLU(),
        )
        self.quantile_fc = nn.Sequential(
            nn.Linear(self.quantile_embed_dim, fc_hidden),
            nn.ReLU(),
        )
        self.head = nn.Linear(fc_hidden, n_actions)

        # Precompute cosine embedding multipliers [0..K-1]
        self.register_buffer(
            "embed_pi_k",
            torch.arange(self.quantile_embed_dim, dtype=torch.float32).view(1, 1, -1) * np.pi,
        )

    def _feature(self, x: torch.Tensor) -> torch.Tensor:
        x = x.float() / 255.0
        x = self.conv(x)
        return self.state_fc(x.flatten(1))  # [B, H]

    def _quantile_embedding(self, taus: torch.Tensor) -> torch.Tensor:
        # taus: [B, N]
        # cosine basis: cos(pi * k * tau)
        cos_emb = torch.cos(taus.unsqueeze(-1) * self.embed_pi_k)  # [B, N, K]
        emb = self.quantile_fc(cos_emb)  # [B, N, H]
        return emb

    def forward(self, x: torch.Tensor, taus: torch.Tensor) -> torch.Tensor:
        # Returns quantile-wise Q: [B, N, A]
        feat = self._feature(x)  # [B, H]
        qemb = self._quantile_embedding(taus)  # [B, N, H]
        fused = feat.unsqueeze(1) * qemb  # [B, N, H]
        out = self.head(fused)  # [B, N, A]
        return out

    @torch.no_grad()
    def expected_q(self, x: torch.Tensor, num_quantiles: int = 32) -> torch.Tensor:
        # Sample taus ~ U(0,1), compute mean across quantiles → [B, A]
        B = x.shape[0]
        device = x.device
        taus = torch.rand(B, int(num_quantiles), device=device)
        q_vals = self.forward(x, taus)  # [B, N, A]
        return q_vals.mean(dim=1)
