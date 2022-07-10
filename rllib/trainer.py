from typing import List

from gym import Env
from tqdm import trange

from rllib._base import _BaseAgent


class Trainer:
    """
    Class to train agent in environment.
    """

    def __init__(
        self,
        env: Env,
    ):
        """
        Trainer initialization with gym environment.

        Args:
            env (Env): gym environment.
        """

        self.env = env

    def train(
        self,
        agent: _BaseAgent,
        n_sessions: int,
        epsilon_decay: float = 0.99,
        t_max: int = 10**4,
    ) -> List[float]:
        """
        Train loop over sessions.

        Args:
            agent (_BaseAgent): RL agent.
            n_sessions (int): number of sessions to train.
            epsilon_decay (float, optional): epsilon decay. Defaults to 0.99.
            t_max (int, optional): max number of one session actions. Defaults to 10**4.

        Returns:
            List[float]: rewards over sessions.
        """

        rewards = []
        for _ in trange(n_sessions, desc="loop over sessions"):
            r = self._play_session(
                agent=agent,
                t_max=t_max,
            )
            rewards.append(r)

            agent.epsilon *= epsilon_decay  # type: ignore

        return rewards

    def _play_session(
        self,
        agent: _BaseAgent,
        t_max: int,
    ) -> float:
        """
        One session cycle.

        Args:
            agent (_BaseAgent): RL agent.
            t_max (int): max number of one session actions.

        Returns:
            float: session reward.
        """

        total_reward = 0.0
        s = self.env.reset()

        for _ in range(t_max):
            a = agent.get_action(s)
            next_s, r, done, _ = self.env.step(a)

            agent.update(s, a, r, next_s)

            s = next_s
            total_reward += r
            if done:
                break

        return total_reward
