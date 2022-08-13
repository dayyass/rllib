import random

import numpy as np
import torch
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

    env.seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def to_one_hot(
    y_tensor: torch.Tensor,
    n_dims: int,
) -> torch.Tensor:
    """
    Helper function that takes an integer vector and convert it to 1-hot matrix.
    """

    device = y_tensor.device

    y_tensor = torch.tensor(
        y_tensor,
        dtype=torch.long,
        device=device,
    ).view(-1, 1)

    y_one_hot = torch.zeros(y_tensor.size()[0], n_dims, device=device).scatter_(
        1, y_tensor, 1
    )

    return y_one_hot


def where(
    cond: torch.Tensor,
    x_1: torch.Tensor,
    x_2: torch.Tensor,
) -> torch.Tensor:
    """
    Helper function like np.where but in torch.
    """

    return (cond * x_1) + ((1 - cond) * x_2)
