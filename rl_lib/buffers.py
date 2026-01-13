"""Experience replay buffers."""

import numpy as np


# =========================
# Rollout Buffer (for PPO)
# =========================
class RolloutBuffer:
    """On-policy rollout buffer for PPO.

    Stores trajectories collected by the current policy.
    """

    def __init__(self, num_envs, buffer_size, obs_shape, action_dim):
        self.num_envs = num_envs
        self.buffer_size = buffer_size
        self.obs_shape = obs_shape
        self.action_dim = action_dim

        # Pre-allocate arrays
        self.obs = np.zeros((buffer_size, num_envs, *obs_shape), dtype=np.float32)
        self.actions = np.zeros((buffer_size, num_envs, action_dim), dtype=np.float32)
        self.rewards = np.zeros((buffer_size, num_envs), dtype=np.float32)
        self.values = np.zeros((buffer_size, num_envs), dtype=np.float32)
        self.log_probs = np.zeros((buffer_size, num_envs), dtype=np.float32)
        self.dones = np.zeros((buffer_size, num_envs), dtype=np.float32)

        self.ptr = 0
        self.num_filled = 0

    def add(self, obs, actions, rewards, values, log_probs, dones):
        """Add a step of transitions."""
        self.obs[self.ptr] = obs
        self.actions[self.ptr] = actions
        self.rewards[self.ptr] = rewards
        self.values[self.ptr] = values
        self.log_probs[self.ptr] = log_probs
        self.dones[self.ptr] = dones

        self.ptr = (self.ptr + 1) % self.buffer_size
        self.num_filled = min(self.num_filled + 1, self.buffer_size)

    def get(self):
        """Get all stored data and reset."""
        # Return data in shape [buffer_size * num_envs, ...]
        return {
            "obs": self.obs[:self.num_filled].reshape(-1, *self.obs_shape),
            "actions": self.actions[:self.num_filled].reshape(-1, self.action_dim),
            "rewards": self.rewards[:self.num_filled].reshape(-1),
            "values": self.values[:self.num_filled].reshape(-1),
            "log_probs": self.log_probs[:self.num_filled].reshape(-1),
            "dones": self.dones[:self.num_filled].reshape(-1),
        }

    def reset(self):
        """Reset the buffer."""
        self.ptr = 0
        self.num_filled = 0


# =========================
# Replay Buffer (for SAC)
# =========================
class ReplayBuffer:
    """Memory-efficient NumPy ring replay buffer."""

    def __init__(self, obs_shape, size):
        self.size = int(size)
        self.obs = np.empty((self.size, *obs_shape), np.uint8)
        self.next = np.empty_like(self.obs)
        self.act = np.empty(self.size, np.int32)
        self.rew = np.empty(self.size, np.float32)
        self.done = np.empty(self.size, np.bool_)
        self.ptr = 0
        self.count = 0

    def push_batch(self, obs, act, rew, nxt, done):
        n = len(obs)
        for i in range(n):
            self.obs[self.ptr] = obs[i]
            self.next[self.ptr] = nxt[i]
            self.act[self.ptr] = act[i]
            self.rew[self.ptr] = rew[i]
            self.done[self.ptr] = done[i]
            self.ptr = (self.ptr + 1) % self.size
            self.count = min(self.count + 1, self.size)

    def sample(self, batch_size):
        idx = np.random.randint(0, self.count, batch_size)
        return (
            self.obs[idx],
            self.act[idx],
            self.rew[idx],
            self.next[idx],
            self.done[idx].astype(np.float32),
        )

    def __len__(self):
        return self.count
