"""Configuration management with inheritance support."""

import os
from pathlib import Path
from typing import Any, Dict
import yaml


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file with inheritance support.

    If the config contains `extends: <path>`, the base config is loaded first
    and then merged with the current config (current config takes precedence).

    Args:
        config_path: Path to YAML config file (relative to project root or absolute)

    Returns:
        Merged configuration dictionary
    """
    config_path = Path(config_path)
    if not config_path.is_absolute():
        # Try relative to current directory, then project root
        if not config_path.exists():
            # Assume it's relative to the project's configs directory
            project_root = Path(__file__).parent.parent
            config_path = project_root / "configs" / config_path

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}

    # Handle inheritance
    if "extends" in config:
        base_path = config.pop("extends")
        # Add .yaml extension if not present
        if not base_path.endswith(".yaml") and not base_path.endswith(".yml"):
            base_path = base_path + ".yaml"
        base_config = load_config(base_path)
        config = merge_configs(base_config, config)

    return config


def merge_configs(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Deep merge two config dictionaries.

    Args:
        base: Base configuration
        override: Override configuration (takes precedence)

    Returns:
        Merged configuration dictionary
    """
    result = base.copy()

    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = merge_configs(result[key], value)
        else:
            result[key] = value

    return result


def save_config(config: Dict[str, Any], path: str) -> None:
    """Save configuration to YAML file.

    Args:
        config: Configuration dictionary
        path: Output file path
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, sort_keys=False, allow_unicode=False)
