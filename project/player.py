import random
import uuid
from enum import Enum
from .base_player import BasePlayer


class PlayerDecision(Enum):
    HIT = 1
    STAY = 2
    DOUBLE_DOWN = 3


class PlayerType(Enum):
    NOOB = 1
    APPREHENSIVE = 2
    AGGRESSIVE = 3
    RANDOM = 4


class Player(BasePlayer):
    player_type: PlayerType
    player_id: str

    def __init__(self, starting_money: int, player_type=PlayerType.RANDOM):
        super().__init__()
        self.money = starting_money
        self.last_hand_res = 0
        self.player_id = str(uuid.uuid4())
        self.player_type = player_type
        self.hands_played = 0

    def make_decision(self, min_bet: int) -> PlayerDecision:
        if self.player_type == PlayerType.NOOB:
            return PlayerDecision.HIT
        if self.player_type == PlayerType.APPREHENSIVE:
            return PlayerDecision.STAY
        if self.player_type == PlayerType.AGGRESSIVE:
            if self.money < min_bet*2:
                return PlayerDecision.HIT

            return PlayerDecision.DOUBLE_DOWN

        return random.choice(list(PlayerDecision))

    def get_money(self) -> int:
        return self.money

    @staticmethod
    def get_bet_amount() -> int:
        return 10

    def submit_bet(self) -> int:
        bet_amount = min(self.get_bet_amount(), self.money)
        self.money = self.money - bet_amount
        self.last_hand_res = -bet_amount
        self.hands_played += 1
        return bet_amount

    def receive_winnings(self, amount: int) -> None:
        self.money += amount
        self.last_hand_res += amount

    def get_last_hand_res(self) -> int:
        return self.last_hand_res

    def get_hands_played(self) -> int:
        return self.hands_played

    def handle_hand_skipped(self) -> None:
        self.last_hand_res = 0

    def type_as_str(self) -> str:
        if self.player_type == PlayerType.NOOB:
            return "NOOB"
        if self.player_type == PlayerType.APPREHENSIVE:
            return "APPREHENSIVE"
        if self.player_type == PlayerType.AGGRESSIVE:
            return "AGGRESSIVE"
        if self.player_type == PlayerType.RANDOM:
            return "RANDOM"
        return "UNKNOWN"
