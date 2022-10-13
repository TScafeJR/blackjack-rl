from enum import Enum
import random
from .base_player import BasePlayer


class PlayerDecision(Enum):
    HIT = 1
    STAY = 2
    DOUBLE_DOWN = 3


class Player(BasePlayer):
    def __init__(self, starting_money: int):
        self.money = starting_money
        super().__init__()

    @staticmethod
    def make_decision() -> PlayerDecision:
        return random.choice(list(PlayerDecision))

    def get_money(self) -> int:
        return self.money

    @staticmethod
    def get_bet_amount() -> int:
        return 10

    def submit_bet(self) -> int:
        bet_amount = min(self.get_bet_amount(), self.money)
        self.money = self.money - bet_amount
        return bet_amount

    def receive_winnings(self, amount: int):
        self.money += amount
