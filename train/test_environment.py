import random
import unittest

from project import (Card, Dealer, HandResult, Observation, Player, PlayerType,
                     TrainTable)
from project.test_table import build_rigged_cards

from .environment import BlackjackEnvironment, encode_observation
from .policies import BasePolicy


class ScriptedPolicy(BasePolicy):
    def __init__(self, action_indices):
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


if __name__ == "__main__":
    unittest.main()
