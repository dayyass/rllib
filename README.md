[![tests](https://github.com/dayyass/rllib/actions/workflows/tests.yml/badge.svg)](https://github.com/dayyass/rllib/actions/workflows/tests.yml)
[![linter](https://github.com/dayyass/rllib/actions/workflows/linter.yml/badge.svg)](https://github.com/dayyass/rllib/actions/workflows/linter.yml)
[![codecov](https://codecov.io/gh/dayyass/rllib/branch/main/graph/badge.svg?token=45O5NRAD8G)](https://codecov.io/gh/dayyass/rllib)

[![python 3.7](https://img.shields.io/badge/python-3.7-blue.svg)](https://github.com/dayyass/rllib#requirements)
[![release (latest by date)](https://img.shields.io/github/v/release/dayyass/rllib)](https://github.com/dayyass/rllib/releases/latest)
[![license](https://img.shields.io/github/license/dayyass/rllib?color=blue)](https://github.com/dayyass/rllib/blob/main/LICENSE)

[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-black)](https://github.com/dayyass/rllib/blob/main/.pre-commit-config.yaml)
[![code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

[![pypi version](https://img.shields.io/pypi/v/pytorch-rllib)](https://pypi.org/project/pytorch-rllib)
[![pypi downloads](https://img.shields.io/pypi/dm/pytorch-rllib)](https://pypi.org/project/pytorch-rllib)

# rllib
Reinforcement Learning Library

## Installation
```
pip install pytorch-rllib
```

## Usage
Implemented agents:
- [ ] CrossEntropy
- [ ] Value / Policy Iteration
- [x] Q-Learning
- [x] Expected Value SARSA
- [ ] DQN
- [ ] Rainbow
- [ ] REINFORCE
- [ ] A2C

```python3
import gym
import numpy as np

from rllib.qlearning import QLearningAgent
from rllib.trainer import Trainer
from rllib.utils import set_global_seed

# make environment
env = gym.make("Taxi-v3")
set_global_seed(seed=42, env=env)

n_actions = env.action_space.n

# make agent
agent = QLearningAgent(
    alpha=0.5,
    epsilon=0.25,
    discount=0.99,
    n_actions=n_actions,
)

# train
trainer = Trainer(env=env)
rewards = trainer.train(
    agent=agent,
    n_sessions=1000,
)

print(f"Mean reward: {np.mean(rewards[-10:])}")  # Mean reward: 8.0
```

More examples you can find [here](https://github.com/dayyass/rllib/tree/main/examples).

## Requirements
Python >= 3.7

## Citation
If you use **rllib** in a scientific publication, we would appreciate references to the following BibTex entry:
```bibtex
@misc{dayyass2022rllib,
    author       = {El-Ayyass, Dani},
    title        = {Reinforcement Learning Library},
    howpublished = {\url{https://github.com/dayyass/rllib}},
    year         = {2022}
}
```
