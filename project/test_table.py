import random
import unittest

from .card import Card
from .dealer import Dealer
from .game import HandResult
from .player import Player, PlayerType
from .table import Table


def build_rigged_cards(draw_order):
    filler = [Card("2", "Hearts") for _ in range(20)]
    return filler + list(reversed(draw_order))


class TestTable(unittest.TestCase):
    def setUp(self):
        random.seed(17)
        self.dealer = Dealer()
        self.table = Table(num_decks=1, minimum_bet=10).add_dealer(self.dealer)

    def rig_deck(self, draw_order):
        self.table.cards.cards = build_rigged_cards(draw_order)
        self.table.cards.discarded_cards = []

    def test_payout_for_result(self):
        self.assertEqual(Table.payout_for_result(HandResult.PLAYER_BLACKJACK, 10), 25)
        self.assertEqual(Table.payout_for_result(HandResult.PLAYER_WIN, 10), 20)
        self.assertEqual(Table.payout_for_result(HandResult.DEALER_BUST, 10), 20)
        self.assertEqual(Table.payout_for_result(HandResult.PUSH, 10), 10)
        self.assertEqual(Table.payout_for_result(HandResult.DEALER_WIN, 10), 0)
        self.assertEqual(Table.payout_for_result(HandResult.BUST, 10), 0)

    def test_resolve_player_result_natural_beats_dealer_21(self):
        player = Player(starting_money=100)
        self.table.add_player(player)
        player.receive_card(Card("A", "Spades"))
        player.receive_card(Card("K", "Hearts"))
        self.dealer.receive_card(Card("7", "Clubs"))
        self.dealer.receive_card(Card("9", "Diamonds"))
        self.dealer.receive_card(Card("5", "Spades"))

        result = self.table.resolve_player_result(player)
        self.assertEqual(result, HandResult.PLAYER_BLACKJACK)

    def test_resolve_player_result_natural_push(self):
        player = Player(starting_money=100)
        self.table.add_player(player)
        player.receive_card(Card("A", "Spades"))
        player.receive_card(Card("K", "Hearts"))
        self.dealer.receive_card(Card("A", "Clubs"))
        self.dealer.receive_card(Card("Q", "Diamonds"))

        result = self.table.resolve_player_result(player)
        self.assertEqual(result, HandResult.PUSH)

    def test_resolve_player_result_dealer_natural_beats_21(self):
        player = Player(starting_money=100)
        self.table.add_player(player)
        player.receive_card(Card("7", "Spades"))
        player.receive_card(Card("9", "Hearts"))
        player.receive_card(Card("5", "Clubs"))
        self.dealer.receive_card(Card("A", "Clubs"))
        self.dealer.receive_card(Card("Q", "Diamonds"))

        result = self.table.resolve_player_result(player)
        self.assertEqual(result, HandResult.DEALER_WIN)

    def test_resolve_player_result_dealer_bust(self):
        player = Player(starting_money=100)
        self.table.add_player(player)
        player.receive_card(Card("10", "Spades"))
        player.receive_card(Card("8", "Hearts"))
        self.dealer.receive_card(Card("10", "Clubs"))
        self.dealer.receive_card(Card("6", "Diamonds"))
        self.dealer.receive_card(Card("K", "Spades"))

        result = self.table.resolve_player_result(player)
        self.assertEqual(result, HandResult.DEALER_BUST)

    def test_dealer_bust_pays_player(self):
        player = Player(starting_money=100, player_type=PlayerType.APPREHENSIVE)
        self.table.add_player(player)
        self.rig_deck(
            [
                Card("10", "Spades"),
                Card("8", "Hearts"),
                Card("10", "Clubs"),
                Card("6", "Diamonds"),
                Card("K", "Spades"),
            ]
        )

        outcomes = self.table.play_hand()

        outcome = outcomes[player.player_id]
        self.assertEqual(outcome.result, HandResult.DEALER_BUST)
        self.assertEqual(outcome.reward, 10)
        self.assertEqual(player.get_money(), 110)

    def test_natural_pays_three_to_two(self):
        player = Player(starting_money=100, player_type=PlayerType.APPREHENSIVE)
        self.table.add_player(player)
        self.rig_deck(
            [
                Card("A", "Spades"),
                Card("K", "Hearts"),
                Card("9", "Clubs"),
                Card("7", "Diamonds"),
                Card("5", "Spades"),
            ]
        )

        outcomes = self.table.play_hand()

        outcome = outcomes[player.player_id]
        self.assertEqual(outcome.result, HandResult.PLAYER_BLACKJACK)
        self.assertEqual(outcome.reward, 15)
        self.assertEqual(player.get_money(), 115)

    def test_play_hand_settles_every_active_player_once(self):
        players = [
            Player(starting_money=100, player_type=PlayerType.APPREHENSIVE),
            Player(starting_money=100, player_type=PlayerType.NOOB),
        ]
        for player in players:
            self.table.add_player(player)

        self.table.play_hand()
        self.assertEqual(self.table.get_hands_played(), 2)

        self.table.play_hand()
        self.assertEqual(self.table.get_hands_played(), 4)

    def test_all_players_face_same_upcard(self):
        players = [
            Player(starting_money=100, player_type=PlayerType.APPREHENSIVE),
            Player(starting_money=100, player_type=PlayerType.APPREHENSIVE),
        ]
        for player in players:
            self.table.add_player(player)
        self.rig_deck(
            [
                Card("10", "Spades"),
                Card("8", "Hearts"),
                Card("9", "Clubs"),
                Card("9", "Hearts"),
                Card("10", "Clubs"),
                Card("7", "Diamonds"),
            ]
        )

        active_players = self.table.start_hand()
        observations = [self.table.build_observation(p) for p in active_players]

        self.assertEqual(len(self.dealer.hand.cards), 2)
        self.assertEqual(observations[0].dealer_upcard_value, 10)
        self.assertEqual(observations[1].dealer_upcard_value, 10)

    def test_money_conservation(self):
        random.seed(7)
        player_types = [
            PlayerType.NOOB,
            PlayerType.APPREHENSIVE,
            PlayerType.AGGRESSIVE,
            PlayerType.RANDOM,
        ]
        table = Table(num_decks=4, minimum_bet=10).add_dealer(Dealer())
        players = []
        for player_type in player_types:
            player = Player(starting_money=200, player_type=player_type)
            players.append(player)
            table.add_player(player)
        table.receive_money(10000)

        total_before = table.get_money() + sum(p.get_money() for p in players)
        for _ in range(100):
            if not table.can_play_hand():
                break
            table.play_hand()
            total_now = table.get_money() + sum(p.get_money() for p in players)
            self.assertEqual(total_now, total_before)
            for player in players:
                self.assertIsInstance(player.get_money(), int)


if __name__ == "__main__":
    unittest.main()
