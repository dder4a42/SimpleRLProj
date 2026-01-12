"""Utility functions for RL training."""

import torch
import yaml


def load_config(config_path="configs/value_based.yaml"):
    """Load configuration from YAML file."""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_device():
    """Get the device to use for training (CUDA if available, else CPU)."""
    return "cuda" if torch.cuda.is_available() else "cpu"
