from __future__ import annotations

from typing import Dict, List, Optional, Self

from .base_table import BaseTable
from .game import HandOutcome, PendingTurn
from .player import Player, PlayerDecision


class TrainTable(BaseTable):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.rebuy = kwargs.get("rebuy", True)
        self.starting_money: Dict[str, int] = {}
        self.rebuys: Dict[str, int] = {}
        self.turn_order: List[Player] = []
        self.turn_index = 0
        self.hand_in_progress = False

    def add_player(self, new_player: Player) -> Self:
        self.starting_money[new_player.player_id] = new_player.get_money()
        self.rebuys[new_player.player_id] = 0
        return super().add_player(new_player)

    def handle_rebuys(self) -> None:
        if not self.rebuy:
            return
        for player in self.players:
            if player.get_money() < self.minimum_bet:
                player.money = self.starting_money[player.player_id]
                self.rebuys[player.player_id] += 1

    def begin_hand(self) -> Self:
        self.handle_rebuys()
        self.turn_order = self.start_hand()
        self.turn_index = 0
        self.hand_in_progress = True
        self.advance_turn()
        return self

    def advance_turn(self) -> None:
        while self.turn_index < len(self.turn_order):
            if self.turn_order[self.turn_index].playing:
                return
            self.turn_index += 1

    def get_pending_turn(self) -> Optional[PendingTurn]:
        if not self.hand_in_progress or self.turn_index >= len(self.turn_order):
            return None
        player = self.turn_order[self.turn_index]
        return PendingTurn(
            player_id=player.player_id,
            observation=self.build_observation(player),
            legal_actions=self.get_legal_actions(player),
        )

    def apply_decision(self, player_id: str, decision: PlayerDecision) -> Self:
        pending_turn = self.get_pending_turn()
        if pending_turn is None or pending_turn.player_id != player_id:
            raise Exception("No pending turn for player")
        player = self.turn_order[self.turn_index]
        self.apply_player_decision(player, decision)
        self.advance_turn()
        return self

    def is_hand_complete(self) -> bool:
        return self.hand_in_progress and self.turn_index >= len(self.turn_order)

    def settle_hand(self) -> Dict[str, HandOutcome]:
        if not self.is_hand_complete():
            raise Exception("Hand is not complete")
        self.play_dealer_hand()
        outcomes = self.settle_bets(self.turn_order)
        self.hand_in_progress = False
        return outcomes

    def get_rebuys(self, player_id: str) -> int:
        return self.rebuys.get(player_id, 0)
