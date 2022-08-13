import gym
import torch
import numpy as np

from rllib.qlearning import ApproximateQLearningAgent
from rllib.trainer import TrainerTorch as Trainer
from rllib.utils import set_global_seed

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# init environment
env = gym.make("CartPole-v0")
set_global_seed(seed=42, env=env)

n_actions = env.action_space.n
n_state = env.observation_space.shape[0]

# init torch model
model = torch.nn.Sequential()
model.add_module("layer1", torch.nn.Linear(n_state, 128))
model.add_module("relu1", torch.nn.ReLU())
model.add_module("layer2", torch.nn.Linear(128, 64))
model.add_module("relu2", torch.nn.ReLU())
model.add_module("values", torch.nn.Linear(64, n_actions))
model = model.to(device)

# init approximate q-learning agent
approximate_q_learning_agent = ApproximateQLearningAgent(
    model=model,
    alpha=0.5,
    epsilon=0.25,
    discount=0.99,
    n_actions=n_actions,
)

# train
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

trainer = Trainer(env=env)

rewards_approximate_q_learning = trainer.train(
    agent=approximate_q_learning_agent,
    optimizer=optimizer,
    n_epochs=100,
    n_sessions=100,
)

# compare results
print(f"Mean reward: {np.mean(rewards_approximate_q_learning[-10:])}")  # Mean reward: 8.0
