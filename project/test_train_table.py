import random
import unittest

from .card import Card
from .dealer import Dealer
from .game import HandResult
from .player import Player, PlayerDecision, PlayerType
from .test_table import build_rigged_cards
from .train_table import TrainTable


class TestTrainTable(unittest.TestCase):
    def setUp(self):
        random.seed(17)
        self.dealer = Dealer()
        self.player = Player(starting_money=100, player_type=PlayerType.RANDOM)
        self.table = TrainTable(num_decks=1, minimum_bet=10, rebuy=False)
        self.table.add_dealer(self.dealer).add_player(self.player)

    def rig_deck(self, draw_order):
        self.table.cards.cards = build_rigged_cards(draw_order)
        self.table.cards.discarded_cards = []

    def test_begin_hand(self):
        self.rig_deck(
            [
                Card("5", "Spades"),
                Card("6", "Hearts"),
                Card("10", "Clubs"),
                Card("8", "Diamonds"),
            ]
        )

        self.table.begin_hand()
        pending_turn = self.table.get_pending_turn()

        self.assertIsNotNone(pending_turn)
        self.assertEqual(pending_turn.player_id, self.player.player_id)
        self.assertEqual(pending_turn.observation.player_total, 11)
        self.assertEqual(pending_turn.observation.dealer_upcard_value, 10)
        self.assertIn(PlayerDecision.HIT, pending_turn.legal_actions)
        self.assertIn(PlayerDecision.STAY, pending_turn.legal_actions)
        self.assertIn(PlayerDecision.DOUBLE_DOWN, pending_turn.legal_actions)
        self.assertFalse(self.table.is_hand_complete())

    def test_hit_to_21_counts_as_win_not_blackjack(self):
        self.rig_deck(
            [
                Card("5", "Spades"),
                Card("6", "Hearts"),
                Card("10", "Clubs"),
                Card("8", "Diamonds"),
                Card("K", "Spades"),
            ]
        )

        self.table.begin_hand()
        self.table.apply_decision(self.player.player_id, PlayerDecision.HIT)

        pending_turn = self.table.get_pending_turn()
        self.assertIsNotNone(pending_turn)
        self.assertEqual(pending_turn.observation.player_total, 21)
        self.assertNotIn(PlayerDecision.DOUBLE_DOWN, pending_turn.legal_actions)

        self.table.apply_decision(self.player.player_id, PlayerDecision.STAY)
        self.assertTrue(self.table.is_hand_complete())

        outcomes = self.table.settle_hand()
        outcome = outcomes[self.player.player_id]
        self.assertEqual(outcome.result, HandResult.PLAYER_WIN)
        self.assertEqual(outcome.reward, 10)

    def test_double_down_debits_second_stake(self):
        self.rig_deck(
            [
                Card("5", "Spades"),
                Card("6", "Hearts"),
                Card("10", "Clubs"),
                Card("8", "Diamonds"),
                Card("K", "Spades"),
            ]
        )

        self.table.begin_hand()
        self.table.apply_decision(self.player.player_id, PlayerDecision.DOUBLE_DOWN)

        self.assertEqual(self.player.get_money(), 80)
        self.assertEqual(self.table.active_bets[self.player.player_id], 20)
        self.assertTrue(self.table.is_hand_complete())

        outcomes = self.table.settle_hand()
        outcome = outcomes[self.player.player_id]
        self.assertEqual(outcome.result, HandResult.PLAYER_WIN)
        self.assertEqual(outcome.reward, 20)
        self.assertEqual(outcome.bet, 20)
        self.assertEqual(self.player.get_money(), 120)

    def test_natural_skips_pending_turn(self):
        self.rig_deck(
            [
                Card("A", "Spades"),
                Card("K", "Hearts"),
                Card("9", "Clubs"),
                Card("7", "Diamonds"),
                Card("5", "Spades"),
            ]
        )

        self.table.begin_hand()

        self.assertIsNone(self.table.get_pending_turn())
        self.assertTrue(self.table.is_hand_complete())

        outcomes = self.table.settle_hand()
        outcome = outcomes[self.player.player_id]
        self.assertEqual(outcome.result, HandResult.PLAYER_BLACKJACK)
        self.assertEqual(outcome.reward, 15)

    def test_apply_decision_wrong_player_raises(self):
        self.rig_deck(
            [
                Card("5", "Spades"),
                Card("6", "Hearts"),
                Card("10", "Clubs"),
                Card("8", "Diamonds"),
            ]
        )

        self.table.begin_hand()
        with self.assertRaises(Exception):
            self.table.apply_decision("unknown-player", PlayerDecision.HIT)

    def test_settle_hand_before_complete_raises(self):
        self.rig_deck(
            [
                Card("5", "Spades"),
                Card("6", "Hearts"),
                Card("10", "Clubs"),
                Card("8", "Diamonds"),
            ]
        )

        self.table.begin_hand()
        with self.assertRaises(Exception):
            self.table.settle_hand()

    def test_handle_rebuys(self):
        rebuy_player = Player(starting_money=100, player_type=PlayerType.RANDOM)
        rebuy_table = TrainTable(num_decks=1, minimum_bet=10, rebuy=True)
        rebuy_table.add_dealer(Dealer()).add_player(rebuy_player)
        rebuy_player.money = 5

        rebuy_table.begin_hand()

        self.assertEqual(rebuy_table.get_rebuys(rebuy_player.player_id), 1)
        self.assertEqual(rebuy_player.get_money(), 90)

    def test_no_rebuy_leaves_player_out(self):
        self.player.money = 5
        self.table.begin_hand()

        self.assertIsNone(self.table.get_pending_turn())
        self.assertTrue(self.table.is_hand_complete())
        self.assertEqual(self.table.settle_hand(), {})

    def test_counting_player_bets_with_the_count(self):
        counting_player = Player(starting_money=1000, player_type=PlayerType.COUNTING)
        counting_table = TrainTable(num_decks=1, minimum_bet=10, rebuy=False)
        counting_table.add_dealer(Dealer()).add_player(counting_player)
        counting_table.running_count = 3

        counting_table.begin_hand()

        self.assertEqual(
            counting_table.active_bets[counting_player.player_id],
            30,
        )


if __name__ == "__main__":
    unittest.main()
