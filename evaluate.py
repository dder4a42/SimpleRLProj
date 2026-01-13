"""
Model evaluation and comparison tool.

Usage:
    # Evaluate single model
    python evaluate.py --checkpoint model.pt --env ALE/Boxing-v5 --episodes 10

    # Compare multiple models
    python evaluate.py --compare model1.pt model2.pt --env ALE/Boxing-v5 --episodes 100
"""

import argparse
import json
from pathlib import Path
from typing import List, Dict, Any

import numpy as np
import torch
import gymnasium as gym
from tqdm import tqdm

from rl_lib.networks import QNet, IQNQNet
from rl_lib.envs import AtariPreprocess, make_env
from rl_lib.utils import get_device


def load_checkpoint(checkpoint_path: str, device: str) -> Dict[str, Any]:
    """Load a trained model checkpoint.

    Args:
        checkpoint_path: Path to checkpoint file
        device: Device to load model on

    Returns:
        Dictionary with model info and state dict
    """
    ckpt = torch.load(checkpoint_path, map_location=device)
    config = ckpt.get("config", {})
    state_dict = ckpt.get("q_state_dict", ckpt.get("state_dict"))

    if state_dict is None:
        raise ValueError(f"Could not find state_dict in {checkpoint_path}")

    # Determine network type from state dict
    if "quantile_fc.0.weight" in state_dict or any("quantile" in k for k in state_dict.keys()):
        net_class = IQNQNet
        quantile_embed_dim = config.get("iqn", {}).get("embed_dim", 64)
    else:
        net_class = QNet
        quantile_embed_dim = None

    # Get architecture from config
    env_cfg = config.get("env", {})
    net_cfg = config.get("network", {})
    frame_stack = env_cfg.get("frame_stack", 4)
    conv_channels = tuple(net_cfg.get("conv_channels", [32, 64, 64]))
    fc_hidden = net_cfg.get("fc_hidden", 512)

    return {
        "state_dict": state_dict,
        "net_class": net_class,
        "frame_stack": frame_stack,
        "conv_channels": conv_channels,
        "fc_hidden": fc_hidden,
        "quantile_embed_dim": quantile_embed_dim,
        "config": config,
    }


def evaluate_model(
    model_info: Dict[str, Any],
    env_name: str,
    num_episodes: int,
    seed: int = 42,
) -> Dict[str, float]:
    """Evaluate a single model on the given environment.

    Args:
        model_info: Model info from load_checkpoint
        env_name: Environment name
        num_episodes: Number of evaluation episodes
        seed: Random seed

    Returns:
        Dictionary with evaluation metrics
    """
    device = get_device()

    # Create environment
    env = gym.make(env_name)
    env.reset(seed=seed)
    env = AtariPreprocess(env, frame_stack=model_info["frame_stack"])

    # Build network
    net_class = model_info["net_class"]
    n_actions = env.action_space.n

    if net_class == IQNQNet:
        net = IQNQNet(
            n_actions,
            in_ch=model_info["frame_stack"],
            conv_channels=model_info["conv_channels"],
            fc_hidden=model_info["fc_hidden"],
            quantile_embed_dim=model_info["quantile_embed_dim"],
        ).to(device)
    else:
        net = QNet(
            n_actions,
            in_ch=model_info["frame_stack"],
            conv_channels=model_info["conv_channels"],
            fc_hidden=model_info["fc_hidden"],
        ).to(device)

    net.load_state_dict(model_info["state_dict"])
    net.eval()

    # Run evaluation episodes
    episode_rewards = []
    episode_lengths = []

    for _ in tqdm(range(num_episodes), desc=f"Evaluating {env_name}", leave=False):
        obs, _ = env.reset()
        done = False
        truncated = False
        ep_reward = 0
        ep_length = 0

        while not (done or truncated):
            obs_tensor = torch.from_numpy(obs).unsqueeze(0).to(device)

            with torch.no_grad():
                if hasattr(net, "expected_q"):
                    q_vals = net.expected_q(obs_tensor, num_quantiles=32)
                else:
                    q_vals = net(obs_tensor)

            action = q_vals.argmax(dim=1).item()
            obs, reward, done, truncated, _ = env.step(action)
            ep_reward += reward
            ep_length += 1

        episode_rewards.append(ep_reward)
        episode_lengths.append(ep_length)

    env.close()

    return {
        "mean_reward": float(np.mean(episode_rewards)),
        "std_reward": float(np.std(episode_rewards)),
        "max_reward": float(np.max(episode_rewards)),
        "min_reward": float(np.min(episode_rewards)),
        "mean_length": float(np.mean(episode_lengths)),
        "std_length": float(np.std(episode_lengths)),
    }


def compare_tensorboard_logs(log_dirs: List[str]) -> None:
    """Compare TensorBoard logs by extracting episode rewards.

    Args:
        log_dirs: List of log directory paths
    """
    try:
        from tensorboard.backend.event_processing import event_accumulator
    except ImportError:
        print("TensorBoard not installed. Install with: pip install tensorboard")
        return

    print("\n=== TensorBoard Log Comparison ===")

    results = []
    for log_dir in log_dirs:
        log_path = Path(log_dir)
        if not log_path.exists():
            print(f"Warning: {log_dir} does not exist")
            continue

        event_files = list(log_path.glob("**/*.tfevents.*"))
        if not event_files:
            print(f"Warning: No event files found in {log_dir}")
            continue

        try:
            ea = event_accumulator.EventAccumulator(str(log_path))
            ea.Reload()

            if "episode/reward" in ea.Tags()["scalars"]:
                events = ea.Scalars("episode/reward")
                rewards = [e.value for e in events]

                results.append({
                    "log_dir": str(log_dir),
                    "num_episodes": len(rewards),
                    "final_reward": rewards[-1] if rewards else 0,
                    "max_reward": max(rewards) if rewards else 0,
                    "mean_last_100": np.mean(rewards[-100:]) if len(rewards) >= 100 else np.mean(rewards) if rewards else 0,
                })
        except Exception as e:
            print(f"Warning: Could not read {log_dir}: {e}")

    if results:
        print(f"\n{'Log Directory':<40} {'Episodes':<12} {'Final':<10} {'Max':<10} {'Mean(100)':<12}")
        print("-" * 86)
        for r in results:
            print(f"{r['log_dir']:<40} {r['num_episodes']:<12} {r['final_reward']:<10.2f} {r['max_reward']:<10.2f} {r['mean_last_100']:<12.2f}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Evaluate and compare trained RL models")
    parser.add_argument("--checkpoint", type=str, help="Path to model checkpoint")
    parser.add_argument("--compare", nargs="+", help="Multiple checkpoints to compare")
    parser.add_argument("--log-dirs", nargs="+", help="TensorBoard log directories to compare")
    parser.add_argument("--env", type=str, default="ALE/Boxing-v5", help="Environment name")
    parser.add_argument("--episodes", type=int, default=10, help="Number of evaluation episodes")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output", type=str, help="Output JSON file for results")

    args = parser.parse_args()

    # Compare logs if specified
    if args.log_dirs:
        compare_tensorboard_logs(args.log_dirs)

    # Evaluate checkpoints
    if args.compare:
        print(f"\n=== Evaluating {len(args.compare)} model(s) on {args.env} ===")
        print(f"Running {args.episodes} episodes per model...\n")

        results = {}
        for ckpt_path in args.compare:
            if not Path(ckpt_path).exists():
                print(f"Warning: {ckpt_path} does not exist, skipping...")
                continue

            print(f"Loading {ckpt_path}...")
            model_info = load_checkpoint(ckpt_path, get_device())

            metrics = evaluate_model(model_info, args.env, args.episodes, seed=args.seed)

            results[ckpt_path] = metrics
            print(f"  Mean Reward: {metrics['mean_reward']:.2f} ± {metrics['std_reward']:.2f}")
            print(f"  Max Reward: {metrics['max_reward']:.2f}\n")

        # Print comparison
        if len(results) > 1:
            print("=== Comparison Summary ===")
            print(f"\n{'Model':<40} {'Mean Reward':<15} {'Std':<10} {'Max':<10}")
            print("-" * 75)
            for name, metrics in results.items():
                print(f"{Path(name).name:<40} {metrics['mean_reward']:<15.2f} {metrics['std_reward']:<10.2f} {metrics['max_reward']:<10.2f}")

            best_model = max(results.items(), key=lambda x: x[1]["mean_reward"])
            print(f"\nBest model: {Path(best_model[0]).name} (mean reward: {best_model[1]['mean_reward']:.2f})")

        # Save results
        if args.output:
            with open(args.output, "w") as f:
                json.dump(results, f, indent=2)
            print(f"\nResults saved to {args.output}")

    elif args.checkpoint:
        print(f"=== Evaluating {args.checkpoint} on {args.env} ===")
        model_info = load_checkpoint(args.checkpoint, get_device())
        metrics = evaluate_model(model_info, args.env, args.episodes, seed=args.seed)

        print(f"\nResults ({args.episodes} episodes):")
        print(f"  Mean Reward: {metrics['mean_reward']:.2f} ± {metrics['std_reward']:.2f}")
        print(f"  Max Reward: {metrics['max_reward']:.2f}")
        print(f"  Min Reward: {metrics['min_reward']:.2f}")
        print(f"  Mean Length: {metrics['mean_length']:.2f} ± {metrics['std_length']:.2f}")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
