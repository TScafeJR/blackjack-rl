import random
import unittest

from project import (Card, Dealer, HandResult, Observation, Player, PlayerType,
                     TrainTable)
from project.test_table import build_rigged_cards

from .environment import (BET_UNITS, COUNT_FEATURE_COUNT, FEATURE_COUNT,
                          FINAL_BET_SCALE, INITIAL_BET_SCALE,
                          BlackjackEnvironment, encode_observation,
                          feature_count_for)
from .policies import BasePolicy


class ScriptedPolicy(BasePolicy):
    def __init__(self, action_indices):
        super().__init__()
        self.action_indices = list(action_indices)

    def select_action(self, pending_turn, features):
        return self.action_indices.pop(0)


class TestEncodeObservation(unittest.TestCase):
    def test_encode_observation(self):
        observation = Observation(
            player_total=21,
            is_soft=True,
            dealer_upcard_value=11,
            can_double=False,
            money=90,
        )
        self.assertEqual(encode_observation(observation), [1.0, 1.0, 1.0, 0.0])

    def test_encode_observation_ranges(self):
        observation = Observation(
            player_total=4,
            is_soft=False,
            dealer_upcard_value=2,
            can_double=True,
            money=90,
        )
        features = encode_observation(observation)
        for feature in features:
            self.assertGreaterEqual(feature, 0.0)
            self.assertLessEqual(feature, 1.0)

    def test_count_feature_is_opt_in(self):
        observation = Observation(
            player_total=16,
            is_soft=False,
            dealer_upcard_value=10,
            can_double=True,
            money=90,
            true_count=5.0,
        )
        self.assertEqual(len(encode_observation(observation)), FEATURE_COUNT)
        with_count = encode_observation(observation, uses_count=True)
        self.assertEqual(len(with_count), COUNT_FEATURE_COUNT)
        self.assertAlmostEqual(with_count[-1], 0.5, places=10)

    def test_count_feature_clamps_and_signs(self):
        for true_count, expected in [(40.0, 1.0), (-40.0, -1.0), (-10.0, -1.0)]:
            observation = Observation(player_total=16, true_count=true_count)
            features = encode_observation(observation, uses_count=True)
            self.assertAlmostEqual(features[-1], expected, places=10)

    def test_feature_count_for_matches_encoding(self):
        self.assertEqual(feature_count_for(False), FEATURE_COUNT)
        self.assertEqual(feature_count_for(True), COUNT_FEATURE_COUNT)


class TestBlackjackEnvironment(unittest.TestCase):
    def setUp(self):
        random.seed(17)
        self.player = Player(starting_money=100, player_type=PlayerType.RANDOM)
        self.table = TrainTable(num_decks=1, minimum_bet=10, rebuy=False)
        self.table.add_dealer(Dealer()).add_player(self.player)

    def rig_deck(self, draw_order):
        self.table.cards.cards = build_rigged_cards(draw_order)
        self.table.cards.discarded_cards = []

    def build_environment(self, policy):
        return BlackjackEnvironment(
            table=self.table,
            policies={self.player.player_id: policy},
            agent_kinds={self.player.player_id: "dqn"},
        )

    def test_play_hand_records_steps_and_reward(self):
        self.rig_deck(
            [
                Card("5", "Spades"),
                Card("6", "Hearts"),
                Card("10", "Clubs"),
                Card("8", "Diamonds"),
                Card("K", "Spades"),
            ]
        )
        environment = self.build_environment(ScriptedPolicy([0, 1]))

        episodes = environment.play_hand()

        self.assertEqual(len(episodes), 1)
        episode = episodes[0]
        self.assertEqual(episode.agent_kind, "dqn")
        self.assertEqual(len(episode.steps), 2)
        self.assertEqual(episode.steps[0].action_index, 0)
        self.assertEqual(episode.steps[0].legal_action_indices, [0, 1, 2])
        self.assertAlmostEqual(episode.steps[0].features[0], 11 / 21.0, places=10)
        self.assertAlmostEqual(episode.steps[0].features[2], 10 / 11.0, places=10)
        self.assertEqual(episode.steps[1].action_index, 1)
        self.assertEqual(episode.steps[1].legal_action_indices, [0, 1])
        self.assertEqual(episode.outcome.result, HandResult.PLAYER_WIN)
        self.assertAlmostEqual(episode.reward, 1.0, places=10)

    def test_play_hand_natural_has_no_steps(self):
        self.rig_deck(
            [
                Card("A", "Spades"),
                Card("K", "Hearts"),
                Card("9", "Clubs"),
                Card("7", "Diamonds"),
                Card("5", "Spades"),
            ]
        )
        environment = self.build_environment(ScriptedPolicy([]))

        episodes = environment.play_hand()

        episode = episodes[0]
        self.assertEqual(len(episode.steps), 0)
        self.assertEqual(episode.outcome.result, HandResult.PLAYER_BLACKJACK)
        self.assertAlmostEqual(episode.reward, 1.5, places=10)


class TestRewardScale(unittest.TestCase):
    def setUp(self):
        random.seed(17)
        self.player = Player(starting_money=1000, player_type=PlayerType.RANDOM)
        self.table = TrainTable(num_decks=1, minimum_bet=10, rebuy=False)
        self.table.add_dealer(Dealer()).add_player(self.player)
        self.table.cards.cards = build_rigged_cards(
            [
                Card("10", "Spades"),
                Card("6", "Hearts"),
                Card("9", "Clubs"),
                Card("8", "Diamonds"),
                Card("K", "Spades"),
            ]
        )
        self.table.cards.discarded_cards = []

    def build_environment(self, reward_scale):
        return BlackjackEnvironment(
            table=self.table,
            policies={self.player.player_id: ScriptedPolicy([2])},
            agent_kinds={self.player.player_id: "dqn"},
            reward_scale=reward_scale,
        )

    def test_final_bet_scale_hides_the_doubled_stake(self):
        episode = self.build_environment(FINAL_BET_SCALE).play_hand()[0]

        self.assertEqual(episode.outcome.bet, 20)
        self.assertAlmostEqual(episode.reward, -1.0, places=10)

    def test_initial_bet_scale_prices_the_doubled_stake(self):
        episode = self.build_environment(INITIAL_BET_SCALE).play_hand()[0]

        self.assertEqual(episode.outcome.bet, 20)
        self.assertAlmostEqual(episode.reward, -2.0, places=10)

    def test_scales_agree_when_the_hand_is_not_doubled(self):
        rewards = []
        for scale in [FINAL_BET_SCALE, INITIAL_BET_SCALE]:
            self.setUp()
            environment = BlackjackEnvironment(
                table=self.table,
                policies={self.player.player_id: ScriptedPolicy([1])},
                agent_kinds={self.player.player_id: "dqn"},
                reward_scale=scale,
            )
            rewards.append(environment.play_hand()[0].reward)
        self.assertAlmostEqual(rewards[0], rewards[1], places=10)

    def test_default_scale_is_unchanged(self):
        environment = BlackjackEnvironment(table=self.table)
        self.assertEqual(environment.reward_scale, FINAL_BET_SCALE)


class StubBetPolicy:
    def __init__(self, action_index):
        self.action_index = action_index
        self.seen_features = None

    def select_action(self, features):
        self.seen_features = features
        return self.action_index


class TestEnvironmentBetting(unittest.TestCase):
    def setUp(self):
        random.seed(17)
        self.player = Player(starting_money=1000, player_type=PlayerType.RANDOM)
        self.table = TrainTable(num_decks=1, minimum_bet=10, rebuy=False)
        self.table.add_dealer(Dealer()).add_player(self.player)
        self.bet_policy = StubBetPolicy(action_index=3)

    def build_environment(self, policy, with_betting=True):
        return BlackjackEnvironment(
            table=self.table,
            policies={self.player.player_id: policy},
            bet_policies=(
                {self.player.player_id: self.bet_policy} if with_betting else {}
            ),
            agent_kinds={self.player.player_id: "dqn-ramp"},
        )

    def test_bet_policy_sets_the_wager(self):
        environment = self.build_environment(ScriptedPolicy([1]))

        episodes = environment.play_hand()

        self.assertEqual(self.player.bet_units, BET_UNITS[3])
        self.assertEqual(episodes[0].bet_units, BET_UNITS[3])
        self.assertEqual(episodes[0].bet_action_index, 3)
        self.assertEqual(episodes[0].bet_features, self.bet_policy.seen_features)

    def test_bet_reward_is_scaled_in_min_bet_units(self):
        self.table.cards.cards = build_rigged_cards(
            [
                Card("10", "Spades"),
                Card("K", "Hearts"),
                Card("9", "Clubs"),
                Card("7", "Diamonds"),
                Card("10", "Spades"),
            ]
        )
        self.table.cards.discarded_cards = []
        environment = self.build_environment(ScriptedPolicy([1]))

        episode = environment.play_hand()[0]

        self.assertEqual(episode.outcome.result, HandResult.DEALER_BUST)
        self.assertEqual(episode.outcome.bet, 40)
        self.assertAlmostEqual(episode.reward, 1.0, places=10)
        self.assertAlmostEqual(episode.bet_reward, 4.0, places=10)

    def test_clamped_wager_is_credited_to_the_level_actually_played(self):
        self.player.money = 30
        environment = self.build_environment(ScriptedPolicy([1]))

        episode = environment.play_hand()[0]

        self.assertEqual(episode.bet_units, 3)
        self.assertEqual(episode.bet_action_index, BET_UNITS.index(3))

    def test_no_bet_policy_leaves_flat_wager(self):
        environment = self.build_environment(ScriptedPolicy([1]), with_betting=False)

        episode = environment.play_hand()[0]

        self.assertEqual(episode.bet_units, 1)
        self.assertIsNone(episode.bet_features)


if __name__ == "__main__":
    unittest.main()
