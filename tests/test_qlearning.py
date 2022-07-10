import unittest

import numpy as np

from examples.Taxi_v3 import rewards_ev_sarsa, rewards_q_learning


class TestQLearning(unittest.TestCase):
    """
    Class for testing Q-Learning Agents.
    """

    def test_q_learning_agent(self):
        self.assertEqual(np.mean(rewards_q_learning[-10:]), 8.0)

    def ev_sarsa_agent(self):
        self.assertEqual(np.mean(rewards_ev_sarsa[-10:]), 7.6)


if __name__ == "__main__":
    unittest.main()
