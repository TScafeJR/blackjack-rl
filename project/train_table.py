from .base_table import BaseTable
from .game import DecisionInfo


class TrainTable(BaseTable):
    def __init__(self, num_decks: int, minimum_bet: int):
        super().__init__(num_decks, minimum_bet)

    def play_hand(self):
        base_decision = DecisionInfo(self.minimum_bet, self.maximum_bet)
        while base_decision.stage == TurnStage.SUBMITTING_BET:
            
