"""
Compare TensorBoard logs from multiple runs.

Usage:
    python compare_logs.py logs/run1 logs/run2 logs/run3
"""

import argparse
import sys
from pathlib import Path
from typing import List

import numpy as np


def compare_tensorboard_logs(log_dirs: List[str]) -> None:
    """Compare TensorBoard logs by extracting episode rewards.

    Args:
        log_dirs: List of log directory paths
    """
    try:
        from tensorboard.backend.event_processing import event_accumulator
    except ImportError:
        print("TensorBoard not installed. Install with: pip install tensorboard")
        sys.exit(1)

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

            # Get available scalar tags
            tags = ea.Tags()["scalars"]

            result = {"log_dir": str(log_dir), "tags": tags}

            # Extract episode rewards if available
            if "episode/reward" in tags:
                events = ea.Scalars("episode/reward")
                rewards = [e.value for e in events]
                result["num_episodes"] = len(rewards)
                result["final_reward"] = rewards[-1] if rewards else 0
                result["max_reward"] = max(rewards) if rewards else 0
                result["mean_reward"] = np.mean(rewards) if rewards else 0
                result["mean_last_100"] = np.mean(rewards[-100:]) if len(rewards) >= 100 else result["mean_reward"]

            results.append(result)

        except Exception as e:
            print(f"Warning: Could not read {log_dir}: {e}")

    if not results:
        print("No valid logs found.")
        return

    # Check if we have episode rewards to compare
    if any("num_episodes" in r for r in results):
        print(f"\n{'Log Directory':<40} {'Episodes':<12} {'Mean':<10} {'Final':<10} {'Max':<10}")
        print("-" * 82)
        for r in results:
            if "num_episodes" in r:
                print(f"{r['log_dir']:<40} {r['num_episodes']:<12} {r['mean_reward']:<10.2f} {r['final_reward']:<10.2f} {r['max_reward']:<10.2f}")
            else:
                print(f"{r['log_dir']:<40} {'(no episode/reward)':<42}")
    else:
        print("\nNo episode/reward data found in logs.")
        print("\nAvailable scalar tags:")
        for r in results:
            print(f"\n{r['log_dir']}:")
            for tag in r["tags"]:
                print(f"  - {tag}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Compare TensorBoard logs from multiple runs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "log_dirs",
        nargs="+",
        help="TensorBoard log directories to compare",
    )

    args = parser.parse_args()

    if not args.log_dirs:
        parser.print_help()
        sys.exit(1)

    compare_tensorboard_logs(args.log_dirs)


if __name__ == "__main__":
    main()
