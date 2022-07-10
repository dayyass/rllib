import random

import numpy as np
from gym import Env


def set_global_seed(
    seed: int,
    env: Env,
):
    """
    Set global seed for reproducibility.

    Args:
        seed (int): seed.
        env (Env): gym environment.
    """

    random.seed(seed)
    np.random.seed(seed)
    env.seed(seed)
