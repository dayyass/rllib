import gym
import numpy as np

from rllib.qlearning import EVSarsaAgent, QLearningAgent
from rllib.trainer import Trainer
from rllib.utils import set_global_seed

# init environment
env = gym.make("Taxi-v3")
set_global_seed(seed=42, env=env)

n_actions = env.action_space.n

# init q-learning agent
agent_q_learning = QLearningAgent(
    alpha=0.5,
    epsilon=0.25,
    discount=0.99,
    n_actions=n_actions,
)

# init expected value sarsa agent
agent_ev_sarsa = EVSarsaAgent(
    alpha=0.5,
    epsilon=0.25,
    discount=0.99,
    n_actions=n_actions,
)

# train
trainer = Trainer(env=env)

train_mean_rewards_q_learning = trainer.train(
    agent=agent_q_learning,
    n_epochs=1000,
)

train_mean_rewards_ev_sarsa = trainer.train(
    agent=agent_ev_sarsa,
    n_epochs=1000,
)

# compare results
print(
    f"Mean train reward: {np.mean(train_mean_rewards_q_learning[-10:])}"
)  # reward: 8.0
print(f"Mean train reward: {np.mean(train_mean_rewards_ev_sarsa[-10:])}")  # reward: 7.6

# inference
inference_reward_q_learning = trainer.play_session(
    agent=agent_q_learning,
    t_max=10**4,
)

inference_reward_ev_sarsa = trainer.play_session(
    agent=agent_ev_sarsa,
    t_max=10**4,
)

# compare results
print(f"Inference reward: {inference_reward_q_learning}")  # reward: 7.0
print(f"Inference reward: {inference_reward_ev_sarsa}")  # reward: 5.0
