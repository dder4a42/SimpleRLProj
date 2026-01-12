"""Neural network architectures for RL."""

import numpy as np
import torch
import torch.nn as nn


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
