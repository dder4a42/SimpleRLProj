"""Load a trained checkpoint, run rollout/evaluation, and report metrics.

Usage examples:
  python run.py --checkpoint-dir checkpoints/ALE_boxing-v5 --episodes 10
  python run.py --checkpoint-dir checkpoints/ALE_boxing-v5 --env ALE/Breakout-v5 --episodes 20
  python run.py --checkpoint-dir checkpoints/ALE_boxing-v5 --checkpoint-file checkpoint_step10000000.pt

Notes:
- Expects a config.yaml alongside the checkpoint file in the same directory.
- Chooses the latest checkpoint_step*.pt if --checkpoint-file is not provided;
  falls back to final_q_model.pt if present.
- Supports pixel or RAM observations based on config env.obs_type.
"""

import argparse
import glob
from pathlib import Path
from typing import Dict, Tuple

import gymnasium as gym
import numpy as np
import torch

from rl_lib.config import load_config
from rl_lib.envs import AtariPreprocess, AtariRamStack
from rl_lib.networks import IQNQNet, IQNQNetRAM, QNet, QNetRAM
from rl_lib.utils import get_device


def _select_checkpoint(ckpt_dir: Path, ckpt_file: str | None) -> Path:
    if ckpt_file:
        path = ckpt_dir / ckpt_file
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint file not found: {path}")
        return path

    # Prefer explicit final model
    final_path = ckpt_dir / "final_q_model.pt"
    if final_path.exists():
        return final_path

    # Otherwise pick the highest step checkpoint
    pattern = str(ckpt_dir / "checkpoint_step*.pt")
    candidates = glob.glob(pattern)
    if not candidates:
        raise FileNotFoundError(f"No checkpoints found in {ckpt_dir}")

    def _step_val(p: str) -> int:
        try:
            return int(Path(p).stem.replace("checkpoint_step", ""))
        except Exception:
            return -1

    best = max(candidates, key=_step_val)
    return Path(best)


def _make_env(env_name: str, env_cfg: Dict, seed: int):
    frameskip = int(env_cfg.get("frameskip", 4))
    repeat_prob = float(env_cfg.get("repeat_action_probability", 0.25))
    frame_stack = int(env_cfg.get("frame_stack", 4))
    obs_type = str(env_cfg.get("obs_type", "pixel")).lower()
    max_episode_steps = int(env_cfg.get("max_episode_steps", 0))

    make_kwargs = {
        "frameskip": frameskip,
        "repeat_action_probability": repeat_prob,
    }
    if obs_type == "ram":
        make_kwargs["obs_type"] = "ram"

    env = gym.make(env_name, **make_kwargs)
    env.reset(seed=seed)

    if max_episode_steps > 0:
        env = gym.wrappers.TimeLimit(env, max_episode_steps=max_episode_steps)

    # Auto-detect RAM spaces to be robust even if obs_type was omitted
    is_ram = obs_type == "ram"
    try:
        is_ram = is_ram or len(getattr(env.observation_space, "shape", ())) == 1
    except Exception:
        pass

    if is_ram:
        env = AtariRamStack(env, frame_stack)
    else:
        env = AtariPreprocess(env, frame_stack)

    return env


def _build_network(config: Dict, obs_shape: Tuple[int, ...], n_actions: int, device: str):
    net_cfg = config.get("network", {})
    conv_channels = tuple(net_cfg.get("conv_channels", [32, 64, 64]))
    fc_hidden = int(net_cfg.get("fc_hidden", 512))
    mlp_hidden = int(net_cfg.get("mlp_hidden", fc_hidden))
    mlp_layers = int(net_cfg.get("mlp_layers", 2))

    method = config.get("training", {}).get("method", "standard").lower()

    if method == "iqn":
        quantile_embed_dim = int(config.get("iqn", {}).get("embed_dim", 64))
        if len(obs_shape) == 3:
            net = IQNQNet(
                n_actions,
                in_ch=obs_shape[0],
                conv_channels=conv_channels,
                fc_hidden=fc_hidden,
                quantile_embed_dim=quantile_embed_dim,
            )
        else:
            input_dim = int(np.prod(obs_shape))
            net = IQNQNetRAM(
                n_actions,
                input_dim=input_dim,
                fc_hidden=mlp_hidden,
                quantile_embed_dim=quantile_embed_dim,
            )
    else:
        if len(obs_shape) == 3:
            net = QNet(
                n_actions,
                in_ch=obs_shape[0],
                conv_channels=conv_channels,
                fc_hidden=fc_hidden,
            )
        else:
            input_dim = int(np.prod(obs_shape))
            net = QNetRAM(
                n_actions,
                input_dim=input_dim,
                hidden_dim=mlp_hidden,
                hidden_layers=mlp_layers,
            )

    return net.to(device), method


def evaluate(ckpt_path: Path, config: Dict, env_name: str, episodes: int, seed: int, device: str):
    ckpt = torch.load(ckpt_path, map_location=device)
    state_dict = ckpt.get("q_state_dict") or ckpt.get("state_dict")
    if state_dict is None:
        raise ValueError(f"Checkpoint missing q_state_dict/state_dict: {ckpt_path}")

    env = _make_env(env_name, config.get("env", {}), seed)
    obs, _ = env.reset()
    obs_shape = obs.shape
    n_actions = env.action_space.n

    net, method = _build_network(config, obs_shape, n_actions, device)
    net.load_state_dict(state_dict)
    net.eval()

    rewards = []
    lengths = []

    for ep in range(episodes):
        obs, _ = env.reset(seed=seed + ep)
        done = False
        truncated = False
        ep_rew = 0.0
        ep_len = 0

        while not (done or truncated):
            obs_t = torch.from_numpy(obs).unsqueeze(0).to(device)
            with torch.no_grad():
                if method == "iqn" and hasattr(net, "expected_q"):
                    q_vals = net.expected_q(obs_t, num_quantiles=config.get("iqn", {}).get("eval_quantiles", 32))
                else:
                    q_vals = net(obs_t)
            action = int(q_vals.argmax(dim=1).item())
            obs, r, done, truncated, _ = env.step(action)
            ep_rew += float(r)
            ep_len += 1

        rewards.append(ep_rew)
        lengths.append(ep_len)

    env.close()

    rewards_arr = np.array(rewards, dtype=np.float32)
    lengths_arr = np.array(lengths, dtype=np.int32)

    return {
        "mean_reward": float(rewards_arr.mean()),
        "std_reward": float(rewards_arr.std()),
        "max_reward": float(rewards_arr.max()),
        "min_reward": float(rewards_arr.min()),
        "mean_length": float(lengths_arr.mean()),
        "std_length": float(lengths_arr.std()),
        "episodes": episodes,
    }


def main():
    parser = argparse.ArgumentParser(description="Rollout a trained value-based checkpoint and report metrics.")
    parser.add_argument("--checkpoint-dir", required=True, help="Directory containing checkpoint and config.yaml")
    parser.add_argument("--checkpoint-file", help="Checkpoint filename (default: latest checkpoint_step*.pt or final_q_model.pt)")
    parser.add_argument("--env", help="Environment name to evaluate (defaults to config env.name)")
    parser.add_argument("--episodes", type=int, default=10, help="Number of evaluation episodes")
    parser.add_argument("--seed", type=int, default=42, help="Base random seed")
    parser.add_argument("--device", choices=["cpu", "cuda"], help="Force device (default: auto)")

    args = parser.parse_args()

    ckpt_dir = Path(args.checkpoint_dir)
    if not ckpt_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory not found: {ckpt_dir}")

    config_path = ckpt_dir / "config.yaml"
    config = load_config(str(config_path))

    env_name = args.env or config.get("env", {}).get("name", "ALE/Boxing-v5")
    device = args.device or get_device()

    ckpt_path = _select_checkpoint(ckpt_dir, args.checkpoint_file)
    print(f"Evaluating checkpoint: {ckpt_path}")
    print(f"Environment: {env_name} | Episodes: {args.episodes} | Device: {device}")

    metrics = evaluate(ckpt_path, config, env_name, args.episodes, args.seed, device)

    print("\n=== Evaluation Summary ===")
    for k, v in metrics.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
