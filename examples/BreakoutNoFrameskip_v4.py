from typing import Tuple

import gym
import numpy as np
import torch
from tqdm import trange
from utils import PrimaryAtariWrap

from rllib.frame_buffer import FrameBuffer
from rllib.qlearning import DQN
from rllib.replay_buffer import ReplayBuffer
from rllib.trainer import TrainerTorchWithReplayBuffer as Trainer
from rllib.utils import set_global_seed

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# init environment
env = gym.make("BreakoutNoFrameskip-v4")
env = FrameBuffer(
    PrimaryAtariWrap(env),
    n_frames=4,
)
set_global_seed(seed=42, env=env)

n_actions = env.action_space.n
state_shape = env.observation_space.shape


# init torch model
class Model(torch.nn.Module):
    def __init__(
        self,
        n_actions: int,
        state_shape: Tuple[int],
    ):
        super().__init__()
        self.n_actions = n_actions
        self.state_shape = state_shape

        self.conv1 = torch.nn.Conv2d(4, 16, kernel_size=3, stride=2)
        self.conv2 = torch.nn.Conv2d(16, 32, kernel_size=3, stride=2)
        self.conv3 = torch.nn.Conv2d(32, 64, kernel_size=3, stride=2)
        self.linear = torch.nn.Linear(3136, 256)
        self.qvalues = torch.nn.Linear(256, n_actions)

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = torch.relu(self.linear(x.view(x.size(0), -1)))
        return self.qvalues(x)


model = Model(
    n_actions=n_actions,
    state_shape=state_shape,
).to(device)

target_network = Model(
    n_actions=n_actions,
    state_shape=state_shape,
).to(device)
target_network.load_state_dict(model.state_dict())

# init agent
agent = DQN(
    model=model,
    target_network=target_network,
    epsilon=0.5,
    discount=0.99,
    n_actions=n_actions,
)

# train
trainer = Trainer(env=env)

# # fill experience replay buffer
exp_replay = ReplayBuffer(size=10**4)

state = trainer.env.reset()
for i in trange(100, desc="fill exp_replay loop"):
    trainer.play_and_record(
        initial_state=state,
        agent=agent,
        exp_replay=exp_replay,
        n_steps=10**2,
    )

print(f"Experience replay buffer size: {len(exp_replay)}")


# # epsilon scheduler
class LinearDecayEpsilonScheduler:
    def __init__(
        self,
        init_epsilon: float,
        final_epsilon: float,
        decay_steps: int,
    ):
        self.init_epsilon = init_epsilon
        self.final_epsilon = final_epsilon
        self.decay_steps = decay_steps

    def __call__(
        self,
        current_step: int,
    ):
        if current_step >= self.decay_steps:
            return self.final_epsilon

        return (
            self.init_epsilon * (self.decay_steps - current_step)
            + self.final_epsilon * current_step
        ) / self.decay_steps


epsilon_scheduler = LinearDecayEpsilonScheduler(
    init_epsilon=1.0,
    final_epsilon=0.3,
    decay_steps=10**6,
)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

train_rewards = trainer.train(
    agent=agent,
    exp_replay=exp_replay,
    optimizer=optimizer,
    n_steps=3 * 10**6,
    batch_size=16,
    transitions_per_step=1,
    refresh_target_network_freq=5000,
    epsilon_scheduler=epsilon_scheduler,
    max_grad_norm=50,
    t_max=10000,
    verbose=True,
    frequency=5000,
)

# train results
print(f"Mean train reward: {np.mean(train_rewards[-10:])}")

# inference
inference_reward = trainer.play_session(
    agent=agent,
    t_max=10**4,
)

# inference results
print(f"Inference reward: {inference_reward}")
