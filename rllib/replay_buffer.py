import random
from collections import namedtuple
from typing import List

import numpy as np

Transition = namedtuple(
    "Transition",
    [
        "state",
        "action",
        "reward",
        "next_state",
        "is_done",
    ],
)

TransitionsBatch = namedtuple(
    "TransitionsBatch",
    [
        "states",
        "actions",
        "rewards",
        "next_states",
        "is_dones",
    ],
)


class ReplayBuffer:
    """
    Replay Buffer.
    """

    def __init__(
        self,
        size: int,
    ):
        """
        Init Replay Buffer.

        Args:
            size (int): Max number of transitions to store in the buffer.
                        When the buffer overflows the old memories are dropped.
        """

        self.size = size

        self.storage: List[Transition] = []
        self._next_idx = 0

    def __len__(self):
        return len(self.storage)

    def __getitem__(
        self,
        idx: int,
    ) -> Transition:
        return self.storage[idx]

    def add(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        is_done: bool,
    ) -> None:
        """
        Add transition into the buffer.
        """

        transition = Transition(
            state=state,
            action=action,
            reward=reward,
            next_state=next_state,
            is_done=is_done,
        )

        if self._next_idx >= len(self.storage):
            self.storage.append(transition)
        else:
            self.storage[self._next_idx] = transition

        self._next_idx = (self._next_idx + 1) % self.size

    def sample(
        self,
        batch_size: int,
    ) -> TransitionsBatch:
        """
        Sample a batch of experience transitions.
        """

        idxes = [random.randint(0, len(self.storage) - 1) for _ in range(batch_size)]
        return self._encode_sample(idxes)

    def _encode_sample(
        self,
        idxes: List[int],
    ) -> TransitionsBatch:
        """
        Encode several sample transitions into batch.
        """

        states, actions, rewards, next_states, is_dones = [], [], [], [], []

        for idx in idxes:
            transition = self.storage[idx]
            state, action, reward, next_state, is_done = transition

            states.append(np.array(state, copy=False))
            actions.append(np.array(action, copy=False))
            rewards.append(reward)
            next_states.append(np.array(next_state, copy=False))
            is_dones.append(is_done)

            transitions_batch = TransitionsBatch(
                states=np.array(states),
                actions=np.array(actions),
                rewards=np.array(rewards),
                next_states=np.array(next_states),
                is_dones=np.array(is_dones),
            )

        return transitions_batch
