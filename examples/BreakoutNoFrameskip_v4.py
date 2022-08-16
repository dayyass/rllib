from typing import Tuple

import atari_wrappers
import gym
import numpy as np
import torch
from framebuffer import FrameBuffer
from gym import Env
from gym.core import ObservationWrapper
from gym.spaces import Box
from PIL import Image
from tqdm import trange

from rllib.qlearning import DQN
from rllib.replay_buffer import ReplayBuffer
from rllib.trainer import TrainerTorchWithReplayBuffer as Trainer  # type: ignore
from rllib.utils import set_global_seed

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# init environment
env = gym.make("BreakoutNoFrameskip-v4")


class PreprocessAtariObs(ObservationWrapper):
    """
    Wrapper that crops, scales image into the desired shapes and grayscales it.
    """

    def __init__(
        self,
        env: Env,
    ):
        super().__init__(env)
        self.img_size = (1, 64, 64)
        self.crop_size = (8, 32, 152, 193)
        self.channel_weights = [0.8, 0.1, 0.1]
        self.observation_space = Box(0.0, 1.0, self.img_size)

    def observation(
        self,
        img: np.ndarray,
    ) -> np.ndarray:
        img = Image.fromarray(img)
        img = img.crop(self.crop_size).resize((self.img_size[1], self.img_size[2]))
        img = np.average(img, weights=self.channel_weights, axis=-1)
        img = np.expand_dims(np.float32(img) / 255, 0)
        return img


def PrimaryAtariWrap(env: Env):
    env = atari_wrappers.MaxAndSkipEnv(env, skip=4)
    env = atari_wrappers.EpisodicLifeEnv(env)
    env = atari_wrappers.FireResetEnv(env)
    env = atari_wrappers.ClipRewardEnv(env)
    env = PreprocessAtariObs(env)
    return env


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
    if len(exp_replay) == 10**4:
        break

print(f"Experience replay buffer size: {len(exp_replay)}")
