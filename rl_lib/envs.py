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


def make_env(env_name, seed, frameskip, repeat_prob, frame_stack):
    """Factory function for creating Atari environments with preprocessing."""
    def thunk():
        env = gym.make(
            env_name,
            frameskip=frameskip,
            repeat_action_probability=repeat_prob,
        )
        env.reset(seed=seed)
        return AtariPreprocess(env, frame_stack)

    return thunk
