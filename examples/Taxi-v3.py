import gym
import numpy as np

import sys
sys.path.append(".")
from rllib.qlearning import QLearningAgent
from rllib.trainer import Trainer
from rllib.utils import set_global_seed

# make environment
env = gym.make("Taxi-v3")
set_global_seed(seed=42, env=env)

n_actions = env.action_space.n

# make agent
agent = QLearningAgent(
    alpha=0.5,
    epsilon=0.25,
    discount=0.99,
    n_actions=n_actions,
)

# train
trainer = Trainer(env=env)
rewards = trainer.train(
    agent=agent,
    n_sessions=1000,
)

print(f"Mean reward: {np.mean(rewards[-10:])}")  # Mean reward: 8.0
