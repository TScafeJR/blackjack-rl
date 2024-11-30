import random
import unittest

from .betting import BetLearner, BetPolicy, build_bet_policy
from .environment import BET_ACTION_COUNT, BET_UNITS, Episode, encode_bet_state


class StubNetwork:
    def __init__(self, values):
        self.values = values

    def forward(self, _batch):
        return [self.values]


class TestEncodeBetState(unittest.TestCase):
    def test_encodes_count_and_penetration(self):
        self.assertEqual(encode_bet_state(0.0, 4.0), [0.0, 0.5])

    def test_clamps_extremes(self):
        for features in [encode_bet_state(40.0, 20.0), encode_bet_state(-40.0, -1.0)]:
            for feature in features:
                self.assertGreaterEqual(feature, -1.0)
                self.assertLessEqual(feature, 1.0)


class TestBetPolicy(unittest.TestCase):
    def test_greedy_pick_takes_highest_value(self):
        policy = BetPolicy(network=StubNetwork([0.0, 0.0, 9.0, 0.0, 0.0]), epsilon=0.0)
        self.assertEqual(policy.units_for([0.0, 0.5]), BET_UNITS[2])

    def test_exploration_stays_in_range(self):
        policy = BetPolicy(
            network=StubNetwork([0.0] * BET_ACTION_COUNT),
            epsilon=1.0,
            rng=random.Random(5),
        )
        for _ in range(50):
            self.assertIn(policy.units_for([0.0, 0.5]), BET_UNITS)

    def test_apply_snapshot_sets_epsilon(self):
        policy = build_bet_policy([8], random.Random(1))
        weights = policy.network.get_weights()
        policy.apply_snapshot({"weights": weights, "epsilon": 0.25})
        self.assertEqual(policy.epsilon, 0.25)


class TestBetLearner(unittest.TestCase):
    def setUp(self):
        self.learner = BetLearner(hidden_sizes=[8], seed=3, batch_size=2)

    def test_skips_hands_without_a_bet_decision(self):
        self.learner.ingest([Episode(player_id="p1")])
        self.assertEqual(len(self.learner.buffer), 0)

    def test_records_one_transition_per_bet(self):
        self.learner.ingest(
            [
                Episode(
                    player_id="p1",
                    bet_features=[0.2, 0.5],
                    bet_action_index=3,
                    bet_reward=-4.0,
                )
            ]
        )
        self.assertEqual(len(self.learner.buffer), 1)

    def test_target_is_the_observed_return(self):
        episode = Episode(
            player_id="p1",
            bet_features=[0.2, 0.5],
            bet_action_index=1,
            bet_reward=2.0,
        )
        self.learner.ingest([episode])
        batch = self.learner.buffer.sample(1)
        self.assertEqual(self.learner.compute_target_values(batch), [2.0])

    def test_network_shape_matches_bet_action_space(self):
        values = self.learner.network.forward([[0.0, 0.5]])[0]
        self.assertEqual(len(values), BET_ACTION_COUNT)


if __name__ == "__main__":
    unittest.main()
