"""Type definitions for RL framework."""

from typing import TypeAlias

import numpy as np
import torch

Action: TypeAlias = int | np.ndarray | torch.Tensor
Observation: TypeAlias = np.ndarray
Reward: TypeAlias = float | np.ndarray
Done: TypeAlias = bool | np.ndarray
LogProb: TypeAlias = torch.Tensor
Value: TypeAlias = torch.Tensor
