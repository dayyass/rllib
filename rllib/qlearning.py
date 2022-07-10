from collections import defaultdict
from typing import Union

import numpy as np

from rllib._base import _BaseAgent


class QLearningAgent(_BaseAgent):
    """
    Q-Learning Agent.
    """

    def __init__(
        self,
        alpha: float,
        epsilon: float,
        discount: float,
        n_actions: int,
    ):
        """
        Q-Learning Agent Initialization.

        Args:
            alpha (float): learning rate.
            epsilon (float): exploration probability.
            discount (float): discount rate (aka gamma).
            n_actions (int): number of possible actions.
        """

        self.alpha = alpha
        self.epsilon = epsilon
        self.discount = discount
        self.n_actions = n_actions

        self._qvalues = defaultdict(lambda: defaultdict(lambda: 0.0))  # type: ignore

    def _get_qvalue(
        self,
        state: int,
        action: int,
    ) -> float:
        """
        Get Q(state, action)
        """

        return self._qvalues[state][action]

    def _set_qvalue(
        self,
        state: int,
        action: int,
        value: float,
    ) -> None:
        """
        Set Q(state, action) = value
        """

        self._qvalues[state][action] = value

    def _get_value(
        self,
        state: int,
    ) -> float:
        """
        Compute agent's estimate of V(s) using current q-values
        V(s) = max_over_action[Q(state, action)] over possible actions.
        """

        possible_actions = range(self.n_actions)

        if len(possible_actions) == 0:
            return 0.0

        value = max([self._get_qvalue(state, action) for action in possible_actions])
        return value

    def get_action(
        self,
        state: int,
    ) -> Union[None, int]:
        """
        Compute action in a state, including exploration.
        """

        possible_actions = range(self.n_actions)

        if len(possible_actions) == 0:
            return None

        action = np.random.choice(
            a=[
                self.get_best_action(state),
                np.random.choice(possible_actions),
            ],
            p=[
                1 - self.epsilon,
                self.epsilon,
            ],
        )

        return int(action)

    def get_best_action(
        self,
        state: int,
    ) -> Union[None, int]:
        """
        Compute the best action to take in a state.
        """

        possible_actions = range(self.n_actions)

        if len(possible_actions) == 0:
            return None

        best_action = np.argmax(
            [self._get_qvalue(state, action) for action in possible_actions]
        )
        return int(best_action)

    def update(
        self,
        state: int,
        action: int,
        reward: float,
        next_state: int,
    ) -> None:
        """
        Q-Value update:
            Q(s,a) := (1 - alpha) * Q(s,a) + alpha * (r + gamma * V(s'))
        """

        q_value = (1 - self.alpha) * self._get_qvalue(state, action) + self.alpha * (
            reward + self.discount * self._get_value(next_state)
        )

        self._set_qvalue(state, action, q_value)


class EVSarsaAgent(QLearningAgent):
    """
    Expected Value SARSA Agent.
    """

    def _get_value(
        self,
        state: int,
    ) -> float:
        """
        Compute agent's estimate of V(s) using current q-values
        V(s) = sum _{over a_i} {pi(a_i | s) * Q(s, a_i)}
        """

        possible_actions = range(self.n_actions)

        if len(possible_actions) == 0:
            return 0.0

        q_values = [self._get_qvalue(state, action) for action in possible_actions]
        max_q_value_idx = np.argmax(q_values)

        value = 0.0

        for i, q_value in enumerate(q_values):
            if i == max_q_value_idx:
                value += (
                    (1 - self.epsilon) + self.epsilon / len(possible_actions)
                ) * q_value
            else:
                value += (self.epsilon / len(possible_actions)) * q_value

        return value
