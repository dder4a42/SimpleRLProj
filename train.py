"""
Unified training script for RL algorithms.

Usage:
    python train.py --algorithm dqn --env ALE/Boxing-v5 --exp-name my-test
    python train.py --algorithm iqn --config configs/iqn.yaml
    python train.py --algorithm dqn --config configs/dqn.yaml --env ALE/Pong-v5

Available algorithms:
    - dqn: Soft Double Q-learning with entropy regularization
    - iqn: Implicit Quantile Networks (distributional RL)
"""

import argparse
import sys

from rl_lib.config import load_config
from rl_lib.algorithms import get_algorithm, ALGORITHMS


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train RL agents on Atari environments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--algorithm", "-a",
        type=str,
        default=None,
        choices=list(ALGORITHMS.keys()),
        help="Algorithm to use",
    )

    parser.add_argument(
        "--config", "-c",
        type=str,
        default=None,
        help="Path to config file (overrides algorithm default)",
    )

    parser.add_argument(
        "--env", "-e",
        type=str,
        default=None,
        help="Override environment name",
    )

    parser.add_argument(
        "--exp-name", "-n",
        type=str,
        default=None,
        help="Experiment name prefix",
    )

    parser.add_argument(
        "--total-steps",
        type=int,
        default=None,
        help="Override total training steps",
    )

    parser.add_argument(
        "--num-envs",
        type=int,
        default=None,
        help="Override number of parallel environments",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Override random seed",
    )

    parser.add_argument(
        "--list-algorithms",
        action="store_true",
        help="List available algorithms and exit",
    )

    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()

    # List algorithms if requested
    if args.list_algorithms:
        print("Available algorithms:")
        for name in sorted(ALGORITHMS.keys()):
            cls = ALGORITHMS[name]
            desc = cls.__doc__.split("\n")[0] if cls.__doc__ else ""
            print(f"  {name:12s} - {desc}")
        return

    # Require algorithm for training
    if args.algorithm is None:
        print("Error: --algorithm is required for training")
        print("\nUse --list-algorithms to see available algorithms.")
        sys.exit(1)

    # Determine config path
    if args.config is None:
        # Use algorithm's default config
        args.config = f"configs/{args.algorithm.lower()}.yaml"

    # Load config
    try:
        config = load_config(args.config)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)

    # Override config with command line arguments
    if args.env:
        config["env"]["name"] = args.env

    if args.total_steps:
        config["training"]["total_steps"] = args.total_steps

    if args.num_envs:
        config["parallel"]["num_envs"] = args.num_envs

    if args.seed is not None:
        config["parallel"]["seed"] = args.seed

    # Set method from algorithm if not in config
    if "method" not in config.get("training", {}):
        config.setdefault("training", {})["method"] = args.algorithm.lower()

    # Set default environment if not specified
    if "name" not in config.get("env", {}):
        config.setdefault("env", {})["name"] = "ALE/Boxing-v5"

    # Get algorithm class and create trainer
    TrainerClass = get_algorithm(args.algorithm)
    trainer = TrainerClass(config, experiment_name=args.exp_name)

    # Run training
    print(f"Starting {args.algorithm.upper()} training on {config['env']['name']}")
    print(f"Config: {args.config}")
    print(f"Experiment: {args.exp_name or 'auto-generated'}")
    print()

    trainer.run()


if __name__ == "__main__":
    main()
