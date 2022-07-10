import gym
import numpy as np

from rllib.qlearning import EVSarsaAgent, QLearningAgent
from rllib.trainer import Trainer
from rllib.utils import set_global_seed

# make environment
env = gym.make("Taxi-v3")
set_global_seed(seed=42, env=env)

n_actions = env.action_space.n

# make q-learning agent
q_learning_agent = QLearningAgent(
    alpha=0.5,
    epsilon=0.25,
    discount=0.99,
    n_actions=n_actions,
)

# make expected value sarsa agent
ev_sarsa_agent = EVSarsaAgent(
    alpha=0.5,
    epsilon=0.25,
    discount=0.99,
    n_actions=n_actions,
)

# train
trainer = Trainer(env=env)

rewards_q_learning = trainer.train(
    agent=q_learning_agent,
    n_sessions=1000,
)

rewards_ev_sarsa = trainer.train(
    agent=ev_sarsa_agent,
    n_sessions=1000,
)

# compare results
print(f"Mean reward: {np.mean(rewards_q_learning[-10:])}")  # Mean reward: 8.0
print(f"Mean reward: {np.mean(rewards_ev_sarsa[-10:])}")  # Mean reward: 7.6
