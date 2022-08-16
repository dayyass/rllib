import numpy as np
from gym import Env
from gym.core import Wrapper
from gym.spaces import Box


class FrameBuffer(Wrapper):
    """
    Frame Buffer.
    """

    def __init__(
        self,
        env: Env,
        n_frames: int = 4,
    ):
        """
        Init Frame Buffer.

        Args:
            env (Env): gym environment.
            n_frames (int, optional): number of frames to handle at a time. Defaults to 4.
        """

        super().__init__(env)

        n_channels, height, width = env.observation_space.shape
        obs_shape = [n_channels * n_frames, height, width]

        self.observation_space = Box(0.0, 1.0, obs_shape)
        self.framebuffer = np.zeros(obs_shape, "float32")

    def reset(self):
        self.framebuffer = np.zeros_like(self.framebuffer)
        self.update_buffer(self.env.reset())
        return self.framebuffer

    def step(self, action):
        new_img, reward, done, info = self.env.step(action)
        self.update_buffer(new_img)
        return self.framebuffer, reward, done, info

    def update_buffer(self, img):
        offset = self.env.observation_space.shape[0]
        axis = 0
        cropped_framebuffer = self.framebuffer[:-offset]
        self.framebuffer = np.concatenate(
            [img, cropped_framebuffer],
            axis=axis,
        )
