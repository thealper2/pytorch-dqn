import random
from typing import Tuple

import numpy as np


class ReplayBuffer:
    """
    Experience replay buffer to store and sample transitions.
    """

    def __init__(self, capacity: int):
        """
        Initialize the replay buffer.

        Args:
            capacity: Maximum capacity of the buffer.
        """
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        """
        Add a transition to the buffer.

        Args:
            state: Current state.
            action: Action taken.
            reward: Reward received.
            next_state: Next state.
            done: Whether the episode ended.
        """
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)

        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size: int) -> Tuple:
        """
        Sample a batch of transitions from the buffer.

        Args:
            batch_size: Number of transitions to sample.

        Returns:
            Batch of transitions.
        """
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*batch)

        return (
            np.array(state),
            np.array(action),
            np.array(reward, dtype=np.float32),
            np.array(next_state),
            np.array(done, dtype=np.float32),
        )

    def __len__(self) -> int:
        """
        Get the current size of the buffer.

        Returns:
            Current size of the buffer.
        """
        return len(self.buffer)
