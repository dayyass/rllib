import atari_wrappers
import numpy as np
from gym import Env
from gym.core import ObservationWrapper
from gym.spaces import Box
from PIL import Image


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


def PrimaryAtariWrap(env: Env) -> Env:
    """
    All Atari Wrappers in one.
    """

    env = atari_wrappers.MaxAndSkipEnv(env, skip=4)
    env = atari_wrappers.EpisodicLifeEnv(env)
    env = atari_wrappers.FireResetEnv(env)
    env = atari_wrappers.ClipRewardEnv(env)
    env = PreprocessAtariObs(env)

    return env
