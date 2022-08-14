import gym
import numpy as np
from gym.core import ObservationWrapper

from rllib.qlearning import QLearningAgent
from rllib.trainer import Trainer
from rllib.utils import set_global_seed

# init environment
env = gym.make("CartPole-v0")


class Binarizer(ObservationWrapper):
    def observation(self, state):
        state[0] = round(state[0], 1)
        state[1] = round(state[1], 1)
        state[2] = round(state[2], 1)
        state[3] = round(state[3], 1)
        return tuple(state)


env = Binarizer(env)
set_global_seed(seed=42, env=env)

n_actions = env.action_space.n

# init agent
agent = QLearningAgent(
    alpha=0.5,
    epsilon=0.5,
    discount=0.99,
    n_actions=n_actions,
)

# train
trainer = Trainer(env=env)

train_rewards = trainer.train(
    agent=agent,
    n_epochs=1000,
)

# train results
print(f"Mean train reward: {np.mean(train_rewards[-10:])}")  # reward: 39.0

# inference
inference_reward = trainer.play_session(
    agent=agent,
    t_max=10**4,
)

# inference results
print(f"Inference reward: {inference_reward}")  # reward: 150.0
