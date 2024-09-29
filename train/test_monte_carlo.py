import unittest

from .environment import Episode, EpisodeStep
from .monte_carlo import MonteCarloLearner


def build_step(action_index):
    return EpisodeStep(
        features=[0.5, 0.0, 0.5, 0.0],
        action_index=action_index,
        legal_action_indices=[0, 1],
    )


class TestMonteCarloLearner(unittest.TestCase):
    def test_handle_episode_discounts_returns(self):
        learner = MonteCarloLearner(seed=1, gamma=0.5)
        episode = Episode(
            player_id="p1",
            agent_kind="mc",
            steps=[build_step(0), build_step(0), build_step(1)],
            reward=1.0,
        )

        learner.ingest([episode])

        rewards = [transition.reward for transition in learner.buffer.transitions]
        self.assertEqual(rewards, [1.0, 0.5, 0.25])
        for transition in learner.buffer.transitions:
            self.assertTrue(transition.done)

    def test_compute_target_values_returns_stored_returns(self):
        learner = MonteCarloLearner(seed=1)
        episode = Episode(
            player_id="p1", agent_kind="mc", steps=[build_step(0)], reward=-1.0
        )
        learner.ingest([episode])

        targets = learner.compute_target_values(learner.buffer.transitions)
        self.assertEqual(targets, [-1.0])


if __name__ == "__main__":
    unittest.main()
