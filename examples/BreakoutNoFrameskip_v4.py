from typing import Tuple

import gym
import torch
from tqdm import trange
from utils import PrimaryAtariWrap

from rllib.framebuffer import FrameBuffer
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
