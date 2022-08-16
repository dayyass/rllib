from typing import Callable, List, Optional, Tuple

import numpy as np
import torch
from gym import Env
from tqdm import trange

from rllib._base import _BaseAgent
from rllib.replay_buffer import ReplayBuffer


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
        n_epochs: int,
        epsilon_decay: float = 0.99,
        t_max: int = 10**4,
        verbose: bool = True,
        frequency: int = 100,
    ) -> List[float]:
        """
        Train loop over epochs.

        Args:
            agent (_BaseAgent): RL agent.
            n_epochs (int): number of epochs to train.
            epsilon_decay (float, optional): epsilon decay. Defaults to 0.99.
            t_max (int, optional): max number of one session actions. Defaults to 10**4.
            verbose (bool, optional): verbose to print. Defaults to True.
            frequency (bool, optional): epochs interval between verbose statements. Defaults to 100.

        Returns:
            List[float]: rewards over epochs.
        """

        rewards = []
        for n_epoch in trange(n_epochs, desc="loop over epochs"):
            reward = self._play_session(
                agent=agent,
                t_max=t_max,
                train=True,
            )
            rewards.append(reward)

            if verbose:
                if (n_epoch + 1) % frequency == 0:
                    print(
                        f"epoch #{n_epoch + 1}\tmean reward = {reward:.3f}\tepsilon = {agent.epsilon:.3f}"  # type: ignore
                    )

            agent.epsilon *= epsilon_decay  # type: ignore

        return rewards

    # TODO: add joblib
    def _play_session(
        self,
        agent: _BaseAgent,
        t_max: int,
        train: bool = False,
    ) -> float:
        """
        One session cycle.

        Args:
            agent (_BaseAgent): RL agent.
            t_max (int): max number of one session actions.
            train (bool, optional): train or inference phase. Defaults to False.

        Returns:
            float: session reward.
        """

        total_reward = 0.0
        s = self.env.reset()

        for _ in range(t_max):
            a = agent.get_action(s)
            next_s, r, done, _ = self.env.step(a)

            if train:
                agent.update(s, a, r, next_s)

            s = next_s
            total_reward += r
            if done:
                break

        return total_reward

    # TODO: allow exploration on/off
    def play_session(
        self,
        agent: _BaseAgent,
        t_max: int,
    ) -> float:
        """
        Play inference session.

        Args:
            agent (_BaseAgent): RL agent.
            t_max (int): max number of one session actions.

        Returns:
            float: session reward.
        """

        return self._play_session(
            agent=agent,
            t_max=t_max,
            train=False,
        )


class TrainerTorch(Trainer):
    """
    Class to train torch agent in environment.
    """

    # TODO: add joblib
    def train(  # type: ignore
        self,
        agent: _BaseAgent,  # TODO: map agents with trainers
        optimizer: torch.optim.Optimizer,
        n_epochs: int,
        n_sessions: int,
        epsilon_decay: float = 0.99,
        t_max: int = 10**4,
        verbose: bool = True,
        frequency: int = 1,
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
            frequency (bool, optional): epochs interval between verbose statements. Defaults to 1.

        Returns:
            List[float]: rewards over epochs.
        """

        rewards = []
        for n_epoch in trange(n_epochs, desc="loop over epochs"):
            session_rewards = []
            for _ in range(n_sessions):
                reward = self._play_session(
                    agent=agent, t_max=t_max, optimizer=optimizer, train=True
                )
                session_rewards.append(reward)

            mean_session_rewards = np.mean(session_rewards)
            rewards.append(mean_session_rewards)

            if verbose:
                if (n_epoch + 1) % frequency == 0:
                    print(
                        f"epoch #{n_epoch + 1}\tmean reward = {mean_session_rewards:.3f}\tepsilon = {agent.epsilon:.3f}"  # type: ignore
                    )

            agent.epsilon *= epsilon_decay  # type: ignore

        return rewards

    def _play_session(  # type: ignore
        self,
        agent: _BaseAgent,
        t_max: int,
        optimizer: Optional[torch.optim.Optimizer] = None,
        train: bool = False,
    ) -> float:
        """
        One session cycle.

        Args:
            agent (_BaseAgent): torch RL agent.
            t_max (int): max number of one session actions.
            optimizer (Optional[torch.optim.Optimizer]): torch optimizer. Defaults to None.
            train (bool, optional): train or inference phase. Defaults to False.

        Returns:
            float: session reward.
        """

        total_reward = 0.0
        s = self.env.reset()

        for _ in range(t_max):
            a = agent.get_action(s)
            next_s, r, done, _ = self.env.step(a)

            if train:
                optimizer.zero_grad()  # type: ignore
                agent.update([s], [a], [r], [next_s], [done]).backward()  # type: ignore
                optimizer.step()  # type: ignore

            s = next_s
            total_reward += r
            if done:
                break

        return total_reward

    # TODO: allow exploration on/off
    def play_session(
        self,
        agent: _BaseAgent,
        t_max: int,
    ) -> float:
        """
        Play inference session.

        Args:
            agent (_BaseAgent): RL agent.
            t_max (int): max number of one session actions.

        Returns:
            float: session reward.
        """

        return self._play_session(
            agent=agent,
            t_max=t_max,
            optimizer=None,
            train=False,
        )


# TODO: maybe inherit from Trainer
# TODO: add _BaseTrainer
class TrainerTorchWithReplayBuffer(TrainerTorch):
    """
    Class to train torch agent in environment with replay buffer.
    """

    def play_and_record(
        self,
        initial_state: np.ndarray,
        agent: _BaseAgent,
        exp_replay: ReplayBuffer,
        n_steps: int,
    ) -> Tuple[float, np.ndarray]:
        """
        Play n_steps in one session and record transition in experience replay buffer.

        Args:
            initial_state (np.ndarray): initial environmant state.
            agent (_BaseAgent): torch RL agent.
            exp_replay (ReplayBuffer): experience replay buffer.
            n_steps (int): number of steps to play in one session.

        Returns:
            Tuple[float, np.ndarray]: session reward and initial state.
        """

        total_reward = 0.0
        s = initial_state

        for _ in range(n_steps):
            a = agent.get_action([s])
            next_s, r, done, _ = self.env.step(a)

            exp_replay.add(
                state=s,
                action=a,
                reward=r,
                next_state=next_s,
                is_done=done,
            )

            s = next_s
            total_reward += r
            if done:
                s = self.env.reset()
                continue

        return total_reward, s

    # TODO: standartize
    # TODO: allow exploration on/off
    def play_session(
        self,
        agent: _BaseAgent,
        t_max: int,
    ) -> float:
        """
        Play inference session.

        Args:
            agent (_BaseAgent): RL agent.
            t_max (int): max number of one session actions.

        Returns:
            float: session reward.
        """

        total_reward = 0.0
        s = self.env.reset()

        for _ in range(t_max):
            a = agent.get_action([s])
            next_s, r, done, _ = self.env.step(a)

            s = next_s
            total_reward += r
            if done:
                break

        return total_reward

    # TODO: add joblib
    def train(  # type: ignore
        self,
        agent: _BaseAgent,  # TODO: map agents with trainers
        exp_replay: ReplayBuffer,
        optimizer: torch.optim.Optimizer,
        n_steps: int,
        batch_size: int,
        transitions_per_step: int,
        refresh_target_network_freq: int,
        epsilon_scheduler: Callable,
        max_grad_norm: float,
        t_max: int,
        verbose: bool = True,
        frequency: int = 1,
    ) -> List[float]:
        """
        Train loop.

        Args:
            agent (_BaseAgent): torch RL agent.
            exp_replay (ReplayBuffer): experience replay buffer.
            optimizer (torch.optim.Optimizer): torch optimizer.
            n_steps (int): number of steps (iterations) to train.
            batch_size (int): number of transitions to sample from exp_replay.
            transitions_per_step (int): number of transitions to play per step.
            refresh_target_network_freq (int): how often update target network weights.
            epsilon_scheduler (Callable): epsilon scheduler that updates agent.epsilon.
            max_grad_norm (Callable): max gradient norm for gradient cliping.
            t_max (int): max number of one inference session actions.
            verbose (bool, optional): verbose to print. Defaults to True.
            frequency (bool, optional): epochs interval between verbose statements. Defaults to 1.

        Returns:
            List[float]: rewards over epochs.
        """

        rewards_list = []
        state = self.env.reset()

        for step in trange(n_steps, desc="loop over steps"):
            agent.epsilon = epsilon_scheduler(step)  # type: ignore

            _, state = self.play_and_record(
                initial_state=state,
                agent=agent,
                exp_replay=exp_replay,
                n_steps=transitions_per_step,
            )

            states, actions, rewards, next_states, is_dones = exp_replay.sample(
                batch_size
            )

            optimizer.zero_grad()
            loss = agent.update(  # type: ignore
                states=states,
                actions=actions,
                rewards=rewards,
                next_states=next_states,
                is_dones=is_dones,
            )
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                agent.model.parameters(),  # type: ignore
                max_norm=max_grad_norm,
            )
            optimizer.step()

            if (step + 1) % refresh_target_network_freq == 0:
                agent.target_network.load_state_dict(agent.model.state_dict())  # type: ignore

            # TODO: add tensorboard
            if verbose:
                if (step + 1) % frequency == 0:
                    print(
                        f"step #{step + 1}\tloss = {loss.item():.3f}\tgrad norm = {grad_norm:.3f}"  # type: ignore
                    )
                    print(
                        f"step #{step + 1}\texp_replay size = {len(exp_replay)}\tepsilon = {agent.epsilon:.3f}"  # type: ignore
                    )
                    reward = self.play_session(
                        agent=agent,
                        t_max=t_max,
                    )
                    print(
                        f"step #{step + 1}\tinference reward = {reward}"  # type: ignore
                    )
                    rewards_list.append(reward)

        return rewards_list
