from __future__ import annotations
from typing import List, Self
import json
from .deck import Deck
from .dealer import Dealer
from .player import Player
from .game import HandResult


class BaseTable:
    def __init__(self, num_decks: int, minimum_bet: int, maximum_bet: int = 100):
        self.minimum_bet = minimum_bet
        self.table_pot = 0
        self.hands_played = 0
        self.cards: Deck = Deck()
        self.dealer = None
        self.players: List[Player] = []
        self.hand_results = {}
        self.card_tracker = {}
        self.handle_init(num_decks)
        self.maximum_bet = maximum_bet

    def handle_init(self, num_decks: int):
        self.add_decks(num_decks)
        self.config_results_map()

    def add_decks(self, num_decks: int) -> None:
        if num_decks != 1:
            for _ in range(num_decks - 1):
                new_deck = Deck()
                self.cards.combine_deck_and_shuffle(new_deck)

    def add_dealer(self, new_dealer: Dealer) -> Self:
        self.dealer = new_dealer
        return self

    def add_player(self, new_player: Player) -> Self:
        self.players.append(new_player)
        return self

    def get_players(self) -> List[Player]:
        return self.players

    def receive_money(self, amount_deposit: int) -> Self:
        self.table_pot += amount_deposit
        return self

    def disperse_winnings(self, winner: Player, amount: int):
        self.table_pot -= amount
        winner.receive_winnings(amount)

    def collect_cards(self):
        if self.dealer is None:
            raise Exception("Dealer is not defined")

        self.dealer.return_cards()

        for player in self.players:
            player.return_cards()

    def track_hand(self, hand_result: HandResult):
        self.hands_played += 1
        self.hand_results[hand_result.name] += 1

    def config_results_map(self):
        for result in HandResult:
            self.hand_results[result.name] = 0

    def get_money(self):
        return self.table_pot

    def get_stats(self):
        return {"hands_played": self.hands_played, "hand_results": self.hand_results}

    def view_stats(self):
        print(f"hands played: {self.hands_played}")
        print(f"hand results: {json.dumps(self.hand_results, indent=4)}")

        # view card distribution / frequency
        print(f"card distribution results: {json.dumps(self.card_tracker, indent=4)}")
        for i, player in enumerate(self.players):
            print(f"player {i} money {player.get_money()}")

        print(f"house money {self.table_pot}")

    def get_results_of_all_player_hands(self) -> List[Player]:
        return self.players

    def get_observation_for_player(self, player):
        return player

    def calculate_reward(self, player, player_turn_result, bet):
        if player_turn_result == HandResult.PLAYER_BLACKJACK:
            return bet * 2.5
        elif player_turn_result == HandResult.PLAYER_WIN:
            return bet * 2
        elif player_turn_result == HandResult.PUSH:
            return 0
        elif player_turn_result == HandResult.DEALER_WIN:
            return -bet
        elif player_turn_result == HandResult.DEALER_BUST:
            return 0
        elif player_turn_result == HandResult.BUST:
            return -bet
        else:
            raise Exception("Invalid hand result")

    @staticmethod
    def player_can_play_hand(player: Player) -> bool:
        return player.get_money() >= 0

    def take_bets(self, deck: Deck) -> dict:
        bets = {}
        for player in self.players:
            if not self.player_can_play_hand(player):
                player.handle_hand_skipped()
                continue
            if self.dealer is None:
                raise Exception("Dealer is not defined")

            self.dealer.deal_player_initial_cards(player, deck)
            player_bet = player.submit_bet()
            self.receive_money(player_bet)
            bets[player.player_id] = player_bet
        return bets

    def any_players_can_play(self) -> bool:
        for p in self.players:
            if p.get_money() > self.minimum_bet:
                return True
        return False

    def can_play_hand(self) -> bool:
        return self.any_players_can_play() and self.get_money() > 0

    def get_hands_played(self) -> int:
        return self.get_stats()['hands_played']
