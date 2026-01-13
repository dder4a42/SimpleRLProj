"""RL library for value-based and policy-based reinforcement learning."""

# Networks
# Algorithms
from rl_lib.algorithms import ALGORITHMS, DQNTrainer, IQNTrainer, get_algorithm

# Base trainer
from rl_lib.base import BaseTrainer

# Buffers
from rl_lib.buffers import ReplayBuffer

# Config
from rl_lib.config import load_config as load_config_with_inheritance
from rl_lib.config import merge_configs, save_config

# Environments
from rl_lib.envs import AtariPreprocess, make_env
from rl_lib.networks import IQNQNet, QNet

# Types
from rl_lib.types import Action, Done, Observation, Reward

# Utils
from rl_lib.utils import get_device, load_config

__all__ = [
    # Networks
    "QNet",
    "IQNQNet",
    # Buffers
    "ReplayBuffer",
    # Environments
    "AtariPreprocess",
    "make_env",
    # Utils
    "load_config",
    "get_device",
    # Config
    "load_config_with_inheritance",
    "save_config",
    "merge_configs",
    # Types
    "Action",
    "Observation",
    "Reward",
    "Done",
    # Base
    "BaseTrainer",
    # Algorithms
    "DQNTrainer",
    "IQNTrainer",
    "get_algorithm",
    "ALGORITHMS",
]
