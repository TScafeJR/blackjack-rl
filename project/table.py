from __future__ import annotations
from typing import List
from .player import Player, PlayerDecision
from .base_table import BaseTable
from .game import HandResult, DecisionInfo, TurnStage


class Table(BaseTable):
    def __init__(self, num_decks: int, minimum_bet: int):
        super().__init__(num_decks, minimum_bet)
        self.turn_stage = TurnStage.UNSPECIFIED
        self.active_bets = {}
        self.active_players = []

    def play_hand(self) -> List[Player]:

        if self.dealer is not None and len(self.players) > 0:
            dealer = self.dealer
            deck = self.cards

            dealer.deal_self_cards(deck)

            for player in self.players:
                if player.get_money() >= self.minimum_bet:
                    self.active_players.append(player)

            for player in self.active_players:
                self.turn_stage = TurnStage.SUBMITTING_BET
                decision = DecisionInfo(self.minimum_bet, self.maximum_bet, self.turn_stage)
                decision_info = player.make_decision(decision)
                player.set_playing(True)
                self.take_player_bet(player, decision_info.bet_amount)
                player_turn_result = HandResult.UNSPECIFIED
                self.dealer.deal_player_initial_cards(player, deck)
                while player.playing:
                    self.turn_stage = TurnStage.PLAYING
                    decision = DecisionInfo(self.minimum_bet, self.maximum_bet, self.turn_stage)
                    decision_info = player.make_decision(decision)
                    player_decision = decision_info.decision
                    if player_decision == PlayerDecision.HIT:
                        dealer.deal_player_card(player, deck)
                        if player.has_bust_hand():
                            player.set_playing(False)
                            player_turn_result = HandResult.BUST
                            break
                    elif player_decision == PlayerDecision.STAY:
                        player.set_playing(False)
                        break
                    elif player_decision == PlayerDecision.DOUBLE_DOWN:
                        dealer.deal_player_card(player, deck)
                        self.active_bets[player.player_id] *= 2
                        double_down_transfer = player.submit_bet()
                        self.receive_money(double_down_transfer)
                        player.set_playing(False)

                        if player.has_bust_hand():
                            player.set_playing(False)
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
                    if dealer.can_hit():
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
                if dealer.has_bust_hand():
                    player_turn_result = HandResult.DEALER_BUST
                    self.track_hand(player_turn_result)
                    continue
                if player_hand_total == dealer_hand_total:
                    player_turn_result = HandResult.PUSH
                    self.disperse_winnings(player, self.active_bets[player.player_id])
                elif player_turn_result == HandResult.PLAYER_BLACKJACK:
                    self.disperse_winnings(player, self.active_bets[player.player_id] * 2.5)
                elif player_hand_total < dealer_hand_total:
                    player_turn_result = HandResult.DEALER_WIN
                elif player_hand_total > dealer_hand_total:
                    player_turn_result = HandResult.PLAYER_WIN
                    self.disperse_winnings(player, self.active_bets[player.player_id] * 2)

                self.track_hand(player_turn_result)

            self.collect_cards()
            return self.get_results_of_all_player_hands()
        return []

    def clear_bets(self):
        self.active_bets = {}

    def any_players_can_play(self) -> bool:
        for p in self.players:
            if p.get_money() > self.minimum_bet:
                return True
        return False

    def can_play_hand(self) -> bool:
        return self.any_players_can_play() and self.get_money() > 0

    def get_hands_played(self) -> int:
        return self.get_stats()['hands_played']

    def take_player_bet(self, player, bet_amount):
        if player.get_money() < bet_amount:
            return 
        self.active_bets[player.player_id] = bet_amount
        self.receive_money(bet_amount)
