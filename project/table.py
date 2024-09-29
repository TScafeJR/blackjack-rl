from __future__ import annotations

from typing import Dict

from .base_table import BaseTable
from .game import DecisionInfo, HandOutcome, TurnStage
from .player import Player


class Table(BaseTable):
    def play_hand(self) -> Dict[str, HandOutcome]:
        if self.dealer is None or len(self.players) == 0:
            return {}

        active_players = self.start_hand()
        for player in active_players:
            self.play_player_turn(player)
        self.play_dealer_hand()
        return self.settle_bets(active_players)

    def play_player_turn(self, player: Player) -> None:
        while player.playing:
            decision_info = DecisionInfo(
                min_bet=self.minimum_bet,
                max_bet=self.maximum_bet,
                stage=TurnStage.PLAYING,
                observation=self.build_observation(player),
                legal_actions=self.get_legal_actions(player),
            )
            decision_result = player.make_decision(decision_info)
            self.apply_player_decision(player, decision_result.decision)
