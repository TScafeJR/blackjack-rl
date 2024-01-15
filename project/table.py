from __future__ import annotations
from .deck import Deck
from .dealer import Dealer
from .player import Player, PlayerDecision
from .error import Errors
from typing import List
from enum import Enum
import json


class HandResult(Enum):
    UNSPECIFIED = "unspecified"
    PUSH = "push"
    BUST = "player bust"
    DEALER_WIN = "dealer win"
    PLAYER_WIN = "player win"
    PLAYER_BLACKJACK = "player blackjack"
    DEALER_BUST = "dealer bust"


class Table:
    BLACKJACK_SCORE = 21

    def __init__(self, num_decks: int, minimum_bet: int):
        self.num_decks = num_decks
        self.minimum_bet = minimum_bet
        self.table_pot = 0
        self.hands_played = 0
        self.cards = []
        self.dealer = None
        self.players: List[Player] = []
        self.hand_results = {}
        self.card_tracker = {}
        self.add_decks()
        self.config_results_map()

    def add_decks(self) -> None:
        table_deck = Deck()

        if self.num_decks != 1:
            for _ in range(self.num_decks - 1):
                new_deck = Deck()
                table_deck.combine_deck_and_shuffle(new_deck)

        self.cards = table_deck

    def add_dealer(self, new_dealer: Dealer) -> Table:
        self.dealer = new_dealer
        return self

    def add_player(self, new_player: Player) -> Table:
        self.players.append(new_player)
        return self

    def get_players(self) -> List[Player]:
        return self.players

    def receive_money(self, amount_deposit: int) -> Table:
        self.table_pot += amount_deposit
        return self

    def disperse_winnings(self, winner: Player, amount: int):
        self.table_pot -= amount
        winner.receive_winnings(amount)

    def collect_cards(self):
        if self.dealer is None:
            Errors.value_undefined("Dealer")
            return

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
        print(f"player money {self.players[0].get_money()}")
        print(f"house money {self.table_pot}")

    def get_results_of_all_player_hands(self) -> List[Player]:
        return self.players

    @staticmethod
    def player_can_play_hand(player: Player) -> bool:
        return player.get_money() >= 0

    def play_hand(self) -> List[int]:
        if self.dealer is not None and len(self.players) > 0:
            dealer = self.dealer
            deck = self.cards

            bets = {}
            for player in self.players:
                if not self.player_can_play_hand(player):
                    player.handle_hand_skipped()
                    continue
                dealer.deal_player_initial_cards(player, deck)
                player_bet = player.submit_bet()
                self.receive_money(player_bet)
                bets[player.player_id] = player_bet

            dealer.deal_self_cards(deck)

            for player in self.players:
                if not self.player_can_play_hand(player):
                    continue
                player_in_turn = True
                player_turn_result = HandResult.UNSPECIFIED

                while player_in_turn:
                    player_hand_total = player.get_hand_value()
                    if player_hand_total == self.BLACKJACK_SCORE:
                        player_turn_result = HandResult.PLAYER_BLACKJACK
                        break

                    player_decision = player.make_decision()

                    if player_decision == PlayerDecision.HIT:
                        dealer.deal_player_card(player, deck)
                        player_hand_total = player.get_hand_value()
                        if player_hand_total > self.BLACKJACK_SCORE:
                            player_in_turn = False
                            player_turn_result = HandResult.BUST
                            break
                    elif player_decision == PlayerDecision.STAY:
                        player_in_turn = False
                        break
                    elif player_decision == PlayerDecision.DOUBLE_DOWN:
                        dealer.deal_player_card(player, deck)
                        bets[player.player_id] *= 2
                        double_down_transfer = player.submit_bet()
                        self.receive_money(double_down_transfer)
                        player_in_turn = False
                        player_hand_total = player.get_hand_value()
                        if player_hand_total > self.BLACKJACK_SCORE:
                            player_turn_result = HandResult.BUST
                        break

                player_hand = player.see_hand()
                for card in player_hand:
                    if card in self.card_tracker:
                        self.card_tracker[card] += 1
                    else:
                        self.card_tracker[card] = 1

                if player_turn_result == HandResult.BUST:
                    self.track_hand(player_turn_result)
                    continue

                dealer_in_turn = True

                while dealer_in_turn:
                    dealer_hand_total = dealer.get_hand_value()

                    if dealer_hand_total < 17:
                        dealer.deal_self_card(deck)
                    else:
                        dealer_in_turn = False
                        break

                dealer_hand = dealer.see_hand()
                for card in dealer_hand:
                    if card in self.card_tracker:
                        self.card_tracker[card] += 1
                    else:
                        self.card_tracker[card] = 1

                player_hand_total = player.get_hand_value()
                dealer_hand_total = dealer.get_hand_value()

                if dealer_hand_total > self.BLACKJACK_SCORE:
                    player_turn_result = HandResult.DEALER_BUST
                    self.track_hand(player_turn_result)
                    continue

                if player_hand_total == dealer_hand_total:
                    player_turn_result = HandResult.PUSH
                    self.disperse_winnings(player, bets[player.player_id])
                elif player_turn_result == HandResult.PLAYER_BLACKJACK:
                    self.disperse_winnings(player, bets[player.player_id] * 2.5)
                elif player_hand_total < dealer_hand_total:
                    player_turn_result = HandResult.DEALER_WIN
                elif player_hand_total > dealer_hand_total:
                    player_turn_result = HandResult.PLAYER_WIN
                    self.disperse_winnings(player, bets[player.player_id] * 2)

                self.track_hand(player_turn_result)

            self.collect_cards()
            return self.get_results_of_all_player_hands()
