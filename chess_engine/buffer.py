# buffer.py
import random
import numpy as np
from collections import deque

class ReplayBuffer:
    def __init__(self, capacity=500_000):
        self.data = deque(maxlen=capacity)

    def push(self, s, p, v):
        self.data.append((s, p, v))

    def sample(self, batch):
        items = random.sample(self.data, batch)
        S, P, V = zip(*items)
        return (np.stack(S),
                np.stack(P),
                np.array(V, dtype=np.float32).reshape(-1, 1))

    def __len__(self):
        return len(self.data)
