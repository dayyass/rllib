from typing import List

import torch
import numpy as np
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

    # TODO: add verbose
    def train(
        self,
        agent: _BaseAgent,
        n_epochs: int,
        epsilon_decay: float = 0.99,
        t_max: int = 10**4,
    ) -> List[float]:
        """
        Train loop over epochs.

        Args:
            agent (_BaseAgent): RL agent.
            n_epochs (int): number of epochs to train.
            epsilon_decay (float, optional): epsilon decay. Defaults to 0.99.
            t_max (int, optional): max number of one session actions. Defaults to 10**4.

        Returns:
            List[float]: rewards over epochs.
        """

        rewards = []
        for _ in trange(n_epochs, desc="loop over epochs"):
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


class TrainerTorch(Trainer):
    """
    Class to train torch agent in environment.
    """

    def train(
        self,
        agent: _BaseAgent,
        optimizer: torch.optim.Optimizer,
        n_epochs: int,
        n_sessions: int,
        epsilon_decay: float = 0.99,
        t_max: int = 10**4,
        verbose: bool = True,
    ) -> List[float]:
        """
        Train loop over epochs.

        Args:
            agent (_BaseAgent): torch RL agent.
            optimizer (torch.optim.Optimizer): torch optimizer.
            n_epochs (int): number of epochs to train.
            n_sessions (int): number of sessions per epoch.
            epsilon_decay (float, optional): epsilon decay. Defaults to 0.99.
            t_max (int, optional): max number of one session actions. Defaults to 10**4.
            verbose (bool, optional): verbose to print. Defaults to True.

        Returns:
            List[float]: rewards over epochs.
        """

        rewards = []
        for n_epoch in trange(n_epochs, desc="loop over epochs"):
            session_rewards = []
            for _ in range(n_sessions):
                reward = self._play_session(
                    agent=agent,
                    optimizer=optimizer,
                    t_max=t_max,
                )
                session_rewards.append(reward)

            mean_session_rewards = np.mean(session_rewards)
            rewards.append(mean_session_rewards)

            if verbose:
                print(f"epoch #{n_epoch}\tmean reward = {mean_session_rewards:.3f}\tepsilon = {agent.epsilon:.3f}")

            agent.epsilon *= epsilon_decay  # type: ignore

        return rewards

    def _play_session(
        self,
        agent: _BaseAgent,
        optimizer: torch.optim.Optimizer,
        t_max: int,
    ) -> float:
        """
        One session cycle.

        Args:
            agent (_BaseAgent): torch RL agent.
            optimizer (torch.optim.Optimizer): torch optimizer.
            t_max (int): max number of one session actions.

        Returns:
            float: session reward.
        """

        total_reward = 0.0
        s = self.env.reset()

        for _ in range(t_max):
            a = agent.get_action(s)
            next_s, r, done, _ = self.env.step(a)

            optimizer.zero_grad()
            agent.update([s], [a], [r], [next_s], [done]).backward()
            optimizer.step()

            s = next_s
            total_reward += r
            if done:
                break

        return total_reward
