import random
import uuid
from enum import Enum
from typing import Self

from .base_player import BasePlayer
from .game import DecisionInfo, TurnStage


class PlayerDecision(Enum):
    UNDEFINED = 0
    HIT = 1
    STAY = 2
    DOUBLE_DOWN = 3
    SUBMIT_BET = 4


class PlayerType(Enum):
    NOOB = 1
    APPREHENSIVE = 2
    AGGRESSIVE = 3
    RANDOM = 4


class DecisionResult:
    def __init__(self, player):
        self.player = player
        self.decision = None
        self.reward = 0
        self.bet_amount = 0

    def set_decision(self, decision: PlayerDecision) -> Self:
        self.decision = decision
        return self

    def set_reward(self, reward: int) -> Self:
        self.reward = reward
        return self

    def set_bet_amount(self, bet_amount: int) -> Self:
        self.bet_amount = bet_amount
        return self


class Player(BasePlayer):
    player_type: PlayerType
    player_id: str

    def __init__(self, **kwargs):
        super().__init__()
        self.money = kwargs.get("starting_money", 0)
        self.last_hand_res = 0
        self.player_id = str(uuid.uuid4())
        self.player_type = kwargs.get("player_type", PlayerType.RANDOM)
        self.hands_played = 0
        self.playing = False

    @staticmethod
    def decide_for_type(
        player_type: PlayerType, decision_info: DecisionInfo
    ) -> PlayerDecision:
        if player_type == PlayerType.NOOB:
            return PlayerDecision.HIT
        if player_type == PlayerType.APPREHENSIVE:
            return PlayerDecision.STAY
        if player_type == PlayerType.AGGRESSIVE:
            observation = decision_info.observation
            money = observation.money if observation is not None else 0
            if money < decision_info.min_bet * 2:
                return PlayerDecision.HIT
            return PlayerDecision.DOUBLE_DOWN
        return random.choice(list(PlayerDecision))

    def set_playing(self, playing: bool) -> None:
        self.playing = playing

    def make_decision(self, decision_info: DecisionInfo) -> DecisionResult:
        decision_result = DecisionResult(self)

        if decision_info.stage == TurnStage.PLAYING:
            return decision_result.set_decision(
                self.decide_for_type(self.player_type, decision_info)
            )

        return decision_result.set_decision(PlayerDecision.UNDEFINED)

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

    def add_to_bet(self, amount: int) -> int:
        extra_amount = min(amount, self.money)
        self.money = self.money - extra_amount
        self.last_hand_res -= extra_amount
        return extra_amount

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
