from __future__ import annotations

import json
import random
from typing import Dict, List, Optional, Self

from .dealer import Dealer
from .deck import Deck
from .game import HandOutcome, HandResult, Observation
from .player import Player, PlayerDecision


class BaseTable:
    def __init__(self, **kwargs):
        self.minimum_bet = kwargs.get("minimum_bet", 10)
        self.maximum_bet = kwargs.get("maximum_bet", 100)
        self.table_pot = 0
        self.hands_played = 0
        self.cards: Deck = Deck()
        self.dealer: Optional[Dealer] = None
        self.players: List[Player] = []
        self.hand_results: Dict[str, int] = {}
        self.card_tracker: Dict[str, int] = {}
        self.active_bets: Dict[str, int] = {}
        self.handle_init(kwargs.get("num_decks", 1))

    def handle_init(self, num_decks: int):
        self.add_decks(num_decks)
        self.config_results_map()

    def add_decks(self, num_decks: int) -> None:
        for _ in range(num_decks - 1):
            self.cards.combine_deck_and_shuffle(Deck())
        self.cards.shuffle_cards()

    def add_dealer(self, new_dealer: Dealer) -> Self:
        self.dealer = new_dealer
        return self

    def add_player(self, new_player: Player) -> Self:
        self.players.append(new_player)
        return self

    def get_players(self) -> List[Player]:
        return self.players

    def get_dealer(self) -> Dealer:
        if self.dealer is None:
            raise Exception("Dealer is not defined")
        return self.dealer

    def receive_money(self, amount_deposit: int) -> Self:
        self.table_pot += amount_deposit
        return self

    def disperse_winnings(self, winner: Player, amount: int):
        self.table_pot -= amount
        winner.receive_winnings(amount)

    def collect_cards(self):
        self.get_dealer().return_cards()

        for player in self.players:
            player.return_cards()

    def track_hand(self, hand_result: HandResult):
        self.hands_played += 1
        self.hand_results[hand_result.name] += 1

    def track_cards(self, shown_cards: List[str]) -> None:
        for card_name in shown_cards:
            self.card_tracker[card_name] = self.card_tracker.get(card_name, 0) + 1

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
        print(f"card distribution results: {json.dumps(self.card_tracker, indent=4)}")
        for i, player in enumerate(self.players):
            print(f"player {i} money {player.get_money()}")

        print(f"house money {self.table_pot}")

    def get_active_players(self) -> List[Player]:
        return [p for p in self.players if p.get_money() >= self.minimum_bet]

    def take_bet(self, player: Player) -> int:
        bet_amount = player.submit_bet()
        self.active_bets[player.player_id] = bet_amount
        self.receive_money(bet_amount)
        return bet_amount

    def deal_initial_cards(self, active_players: List[Player]) -> None:
        dealer = self.get_dealer()
        for player in active_players:
            dealer.deal_player_initial_cards(player, self.cards)
        dealer.deal_self_cards(self.cards)

    def start_hand(self) -> List[Player]:
        self.collect_cards()
        self.active_bets = {}
        active_players = self.get_active_players()
        for player in active_players:
            self.take_bet(player)
            player.set_playing(True)
        self.deal_initial_cards(active_players)
        for player in active_players:
            if player.has_natural():
                player.set_playing(False)
        return active_players

    def can_double(self, player: Player) -> bool:
        current_bet = self.active_bets.get(player.player_id, 0)
        return len(player.hand.cards) == 2 and 0 < current_bet <= player.get_money()

    def build_observation(self, player: Player) -> Observation:
        upcard = self.get_dealer().preview_card()
        return Observation(
            player_total=player.get_hand_value(),
            is_soft=player.has_soft_hand(),
            dealer_upcard_value=max(upcard.get_value()),
            can_double=self.can_double(player),
            money=player.get_money(),
        )

    def get_legal_actions(self, player: Player) -> List[PlayerDecision]:
        legal_actions = [PlayerDecision.HIT, PlayerDecision.STAY]
        if self.can_double(player):
            legal_actions.append(PlayerDecision.DOUBLE_DOWN)
        return legal_actions

    def clamp_decision(
        self, player: Player, decision: PlayerDecision
    ) -> PlayerDecision:
        legal_actions = self.get_legal_actions(player)
        if decision in legal_actions:
            return decision
        return random.choice(legal_actions)

    def apply_player_decision(self, player: Player, decision: PlayerDecision) -> None:
        dealer = self.get_dealer()
        applied_decision = self.clamp_decision(player, decision)

        if applied_decision == PlayerDecision.HIT:
            dealer.deal_player_card(player, self.cards)
            if player.has_bust_hand():
                player.set_playing(False)
            return

        if applied_decision == PlayerDecision.STAY:
            player.set_playing(False)
            return

        if applied_decision == PlayerDecision.DOUBLE_DOWN:
            extra_amount = player.add_to_bet(self.active_bets[player.player_id])
            self.active_bets[player.player_id] += extra_amount
            self.receive_money(extra_amount)
            dealer.deal_player_card(player, self.cards)
            player.set_playing(False)

    def play_dealer_hand(self) -> None:
        dealer = self.get_dealer()
        while dealer.can_hit():
            dealer.deal_self_card(self.cards)

    def resolve_player_result(self, player: Player) -> HandResult:
        dealer = self.get_dealer()

        if player.has_bust_hand():
            return HandResult.BUST
        if player.has_natural():
            return (
                HandResult.PUSH if dealer.has_natural() else HandResult.PLAYER_BLACKJACK
            )
        if dealer.has_natural():
            return HandResult.DEALER_WIN
        if dealer.has_bust_hand():
            return HandResult.DEALER_BUST

        player_total = player.get_hand_value()
        dealer_total = dealer.get_hand_value()
        if player_total == dealer_total:
            return HandResult.PUSH
        return (
            HandResult.PLAYER_WIN
            if player_total > dealer_total
            else HandResult.DEALER_WIN
        )

    @staticmethod
    def payout_for_result(result: HandResult, bet: int) -> int:
        if result == HandResult.PLAYER_BLACKJACK:
            return bet + (bet * 3) // 2
        if result == HandResult.PLAYER_WIN:
            return bet * 2
        if result == HandResult.DEALER_BUST:
            return bet * 2
        if result == HandResult.PUSH:
            return bet
        return 0

    def settle_bets(self, active_players: List[Player]) -> Dict[str, HandOutcome]:
        outcomes = {}
        for player in active_players:
            result = self.resolve_player_result(player)
            payout_amount = self.payout_for_result(
                result, self.active_bets[player.player_id]
            )
            if payout_amount > 0:
                self.disperse_winnings(player, payout_amount)
            self.track_hand(result)
            self.track_cards(player.see_hand())
            outcomes[player.player_id] = HandOutcome(
                result=result,
                reward=player.get_last_hand_res(),
                bet=self.active_bets[player.player_id],
                money_after=player.get_money(),
            )
        self.track_cards(self.get_dealer().see_hand())
        self.collect_cards()
        return outcomes

    def any_players_can_play(self) -> bool:
        for p in self.players:
            if p.get_money() >= self.minimum_bet:
                return True
        return False

    def can_play_hand(self) -> bool:
        return self.any_players_can_play() and self.get_money() > 0

    def get_hands_played(self) -> int:
        return self.hands_played
