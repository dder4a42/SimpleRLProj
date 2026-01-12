"""Experience replay buffers."""

import numpy as np


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
