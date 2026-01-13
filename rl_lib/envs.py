"""Environment wrappers and factories."""

from collections import deque

import cv2
import gymnasium as gym
import numpy as np

# Register ALE environments with gymnasium
try:
    import ale_py
    gym.register_envs(ale_py)
except Exception:
    pass  # ale-py not installed or already registered


# =========================
# Atari Environments
# =========================
class AtariPreprocess(gym.Wrapper):
    """Atari environment preprocessing: grayscale, resize, frame stacking."""

    def __init__(self, env, frame_stack=4):
        super().__init__(env)
        self.frame_stack = frame_stack
        self.frames = deque(maxlen=frame_stack)
        self.observation_space = gym.spaces.Box(0, 255, (frame_stack, 84, 84), np.uint8)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        obs = self._proc(obs)
        self.frames.clear()
        for _ in range(self.frame_stack):
            self.frames.append(obs)
        return np.stack(self.frames), info

    def step(self, action):
        obs, r, term, trunc, info = self.env.step(action)
        obs = self._proc(obs)
        self.frames.append(obs)
        return np.stack(self.frames), r, term, trunc, info

    @staticmethod
    def _proc(obs):
        obs = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        obs = cv2.resize(obs, (84, 84), interpolation=cv2.INTER_AREA)
        return obs


def make_atari_env(env_name, seed, frameskip, repeat_prob, frame_stack, max_episode_steps=None):
    """Factory function for creating Atari environments with preprocessing."""
    def thunk():
        env = gym.make(
            env_name,
            frameskip=frameskip,
            repeat_action_probability=repeat_prob,
        )
        env.reset(seed=seed)

        # Optionally apply a TimeLimit wrapper to ensure episodes terminate
        if max_episode_steps is not None and int(max_episode_steps) > 0:
            env = gym.wrappers.TimeLimit(env, max_episode_steps=int(max_episode_steps))

        return AtariPreprocess(env, frame_stack)

    return thunk


# =========================
# MuJoCo Environments
# =========================
class MuJoCoPreprocess(gym.Wrapper):
    """MuJoCo environment preprocessing (identity wrapper for consistency)."""

    def __init__(self, env):
        super().__init__(env)
        # No preprocessing needed for MuJoCo

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

    def step(self, action):
        return self.env.step(action)


def make_mujoco_env(env_name, seed, max_episode_steps=None):
    """Factory function for creating MuJoCo environments."""
    def thunk():
        env = gym.make(env_name)
        env.reset(seed=seed)
        # MuJoCo can also optionally be time-limited
        if max_episode_steps is not None and int(max_episode_steps) > 0:
            env = gym.wrappers.TimeLimit(env, max_episode_steps=int(max_episode_steps))
        return MuJoCoPreprocess(env)

    return thunk


# =========================
# Generic Factory
# =========================
def is_mujoco_env(env_name: str) -> bool:
    """Check if environment is a MuJoCo environment."""
    mujoco_keywords = [
        "halfcheetah", "hopper", "walker2d", "ant",
        "humanoid", "swimmer", "reacher", "pusher",
        "invertedpendulum", "inverteddoublependulum"
    ]
    env_lower = env_name.lower()
    return any(keyword in env_lower for keyword in mujoco_keywords)


def make_env(env_name, seed, frameskip=4, repeat_prob=0.25, frame_stack=4, max_episode_steps=None):
    """Generic factory function for creating environments.

    Detects environment type and applies appropriate preprocessing.

    Args:
        env_name: Name of the environment
        seed: Random seed
        frameskip: Frame skip (for Atari)
        repeat_prob: Sticky action probability (for Atari)
        frame_stack: Number of frames to stack (for Atari)

    Returns:
        Thunk that creates the environment
    """
    if is_mujoco_env(env_name):
        return make_mujoco_env(env_name, seed, max_episode_steps=max_episode_steps)
    else:
        return make_atari_env(env_name, seed, frameskip, repeat_prob, frame_stack, max_episode_steps=max_episode_steps)
