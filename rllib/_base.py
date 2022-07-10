from abc import ABC, abstractmethod


class _BaseAgent(ABC):
    """
    Base RL Agent.
    """

    @abstractmethod
    def get_action(self, state):
        ...

    @abstractmethod
    def get_best_action(self, state):
        ...

    @abstractmethod
    def update(self, state, action, reward, next_state):
        ...
