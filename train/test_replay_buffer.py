import random
import unittest

from .replay_buffer import ReplayBuffer, Transition


class TestReplayBuffer(unittest.TestCase):
    def test_push_within_capacity(self):
        buffer = ReplayBuffer(capacity=3)
        for index in range(3):
            buffer.push(Transition(reward=float(index)))
        self.assertEqual(len(buffer), 3)

    def test_push_wraps_around_capacity(self):
        buffer = ReplayBuffer(capacity=3)
        for index in range(5):
            buffer.push(Transition(reward=float(index)))

        self.assertEqual(len(buffer), 3)
        rewards = sorted(transition.reward for transition in buffer.transitions)
        self.assertEqual(rewards, [2.0, 3.0, 4.0])

    def test_sample_is_deterministic_under_seed(self):
        buffer_a = ReplayBuffer(capacity=10, rng=random.Random(3))
        buffer_b = ReplayBuffer(capacity=10, rng=random.Random(3))
        for index in range(10):
            buffer_a.push(Transition(reward=float(index)))
            buffer_b.push(Transition(reward=float(index)))

        sample_a = [transition.reward for transition in buffer_a.sample(4)]
        sample_b = [transition.reward for transition in buffer_b.sample(4)]
        self.assertEqual(sample_a, sample_b)


if __name__ == "__main__":
    unittest.main()
