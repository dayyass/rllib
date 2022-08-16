import gym
import numpy as np
import torch

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

# init agent
agent = ApproximateQLearningAgent(
    model=model,
    alpha=0.5,
    epsilon=0.5,
    discount=0.99,
    n_actions=n_actions,
)

# train
trainer = Trainer(env=env)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
train_rewards = trainer.train(
    agent=agent,
    optimizer=optimizer,
    n_epochs=20,
    n_sessions=100,
)

# train results
print(f"Mean train reward: {np.mean(train_rewards[-10:])}")  # reward: 120.318

# inference
inference_reward = trainer.play_session(
    agent=agent,
    t_max=10**4,
)

# inference results
print(f"Inference reward: {inference_reward}")  # reward: 171.0
