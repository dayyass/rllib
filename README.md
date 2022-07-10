# rllib
Reinforcement Learning Library.

## Installation
```
pip install pytorch-rllib
```

## Usage
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

More examples you can find [here](examples).

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
