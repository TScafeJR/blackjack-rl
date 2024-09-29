import unittest

from .dqn import DQNLearner
from .environment import Episode, EpisodeStep
from .replay_buffer import Transition


def build_episode(reward):
    return Episode(
        player_id="p1",
        agent_kind="dqn",
        steps=[
            EpisodeStep(
                features=[0.5, 0.0, 0.5, 1.0],
                action_index=0,
                legal_action_indices=[0, 1, 2],
            ),
            EpisodeStep(
                features=[0.9, 0.0, 0.5, 0.0],
                action_index=1,
                legal_action_indices=[0, 1],
            ),
        ],
        reward=reward,
    )


class TestDQNLearner(unittest.TestCase):
    def setUp(self):
        self.learner = DQNLearner(seed=1, batch_size=4, epsilon_decay_hands=100)

    def test_handle_episode_unrolls_transitions(self):
        self.learner.ingest([build_episode(1.0)])

        self.assertEqual(len(self.learner.buffer), 2)
        first = self.learner.buffer.transitions[0]
        last = self.learner.buffer.transitions[1]
        self.assertFalse(first.done)
        self.assertEqual(first.reward, 0.0)
        self.assertEqual(first.next_features, [0.9, 0.0, 0.5, 0.0])
        self.assertEqual(first.next_legal_action_indices, [0, 1])
        self.assertTrue(last.done)
        self.assertEqual(last.reward, 1.0)

    def test_compute_target_values_terminal(self):
        transition = Transition(
            features=[0.5, 0.0, 0.5, 1.0], action_index=0, reward=1.5, done=True
        )
        self.assertEqual(self.learner.compute_target_values([transition]), [1.5])

    def test_compute_target_values_bootstraps_legal_max(self):
        next_features = [0.9, 0.0, 0.5, 0.0]
        transition = Transition(
            features=[0.5, 0.0, 0.5, 1.0],
            action_index=0,
            reward=0.0,
            next_features=next_features,
            next_legal_action_indices=[1],
            done=False,
        )

        next_q_values = self.learner.target_network.forward([next_features])[0]
        expected = self.learner.gamma * next_q_values[1]

        target_values = self.learner.compute_target_values([transition])
        self.assertAlmostEqual(target_values[0], expected, places=10)

    def test_train_step_requires_full_batch(self):
        self.assertIsNone(self.learner.train_step())

    def test_train_step_updates_and_syncs_target(self):
        learner = DQNLearner(
            seed=1, batch_size=2, target_sync_interval=1, epsilon_decay_hands=100
        )
        learner.ingest([build_episode(1.0)])

        loss_value = learner.train_step()

        self.assertIsNotNone(loss_value)
        self.assertEqual(learner.train_steps, 1)
        self.assertEqual(
            learner.target_network.get_weights(), learner.network.get_weights()
        )

    def test_epsilon_decays_linearly(self):
        self.assertEqual(self.learner.get_epsilon(), 1.0)
        self.learner.hands_seen = 50
        self.assertAlmostEqual(self.learner.get_epsilon(), 0.525, places=10)
        self.learner.hands_seen = 200
        self.assertEqual(self.learner.get_epsilon(), 0.05)


if __name__ == "__main__":
    unittest.main()
