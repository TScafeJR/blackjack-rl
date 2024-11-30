import random
import uuid
from enum import Enum
from typing import Optional, Self

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
    BASIC = 5
    COUNTING = 6


BET_UNIT_CAP = 5
DEFAULT_BASE_BET = 10


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
        self.true_count = 0.0
        self.base_bet = kwargs.get("base_bet", DEFAULT_BASE_BET)
        self.bet_units = 1

    @staticmethod
    def should_double_hard(observation) -> bool:
        total = observation.player_total
        upcard = observation.dealer_upcard_value
        if not observation.can_double:
            return False
        if total == 11:
            return True
        if total == 10:
            return upcard <= 9
        return total == 9 and 3 <= upcard <= 6

    @staticmethod
    def decide_basic_hard(observation) -> PlayerDecision:
        total = observation.player_total
        upcard = observation.dealer_upcard_value
        if Player.should_double_hard(observation):
            return PlayerDecision.DOUBLE_DOWN
        if total >= 17:
            return PlayerDecision.STAY
        if total >= 13:
            return PlayerDecision.STAY if upcard <= 6 else PlayerDecision.HIT
        if total == 12:
            return PlayerDecision.STAY if 4 <= upcard <= 6 else PlayerDecision.HIT
        return PlayerDecision.HIT

    @staticmethod
    def should_double_soft(observation) -> bool:
        total = observation.player_total
        upcard = observation.dealer_upcard_value
        if not observation.can_double:
            return False
        if total == 19:
            return upcard == 6
        if total == 18:
            return upcard <= 6
        if total == 17:
            return 3 <= upcard <= 6
        if total in (15, 16):
            return 4 <= upcard <= 6
        return total in (13, 14) and 5 <= upcard <= 6

    @staticmethod
    def decide_basic_soft(observation) -> PlayerDecision:
        total = observation.player_total
        upcard = observation.dealer_upcard_value
        if Player.should_double_soft(observation):
            return PlayerDecision.DOUBLE_DOWN
        if total >= 19:
            return PlayerDecision.STAY
        if total == 18:
            return PlayerDecision.STAY if upcard <= 8 else PlayerDecision.HIT
        return PlayerDecision.HIT

    @staticmethod
    def decide_basic_strategy(observation) -> PlayerDecision:
        if observation is None:
            return PlayerDecision.HIT
        if observation.is_soft:
            return Player.decide_basic_soft(observation)
        return Player.decide_basic_hard(observation)

    COUNTING_DOUBLE_DEVIATIONS = [
        (10, (10, 11), 4.0),
        (9, (2,), 1.0),
        (9, (7,), 3.0),
        (8, (6,), 2.0),
    ]
    COUNTING_STAND_DEVIATIONS = [
        (16, (10,), 1.0),
        (15, (10,), 4.0),
        (12, (2,), 3.0),
        (12, (3,), 2.0),
    ]
    COUNTING_HIT_DEVIATIONS = [
        (13, (2,), -1.0),
        (12, (4,), -0.5),
    ]

    @staticmethod
    def counting_deviation(observation) -> Optional[PlayerDecision]:
        if observation.is_soft:
            return None
        total = observation.player_total
        upcard = observation.dealer_upcard_value
        true_count = observation.true_count
        if observation.can_double:
            for rule_total, upcards, floor in Player.COUNTING_DOUBLE_DEVIATIONS:
                if total == rule_total and upcard in upcards and true_count >= floor:
                    return PlayerDecision.DOUBLE_DOWN
        for rule_total, upcards, floor in Player.COUNTING_STAND_DEVIATIONS:
            if total == rule_total and upcard in upcards and true_count >= floor:
                return PlayerDecision.STAY
        for rule_total, upcards, ceiling in Player.COUNTING_HIT_DEVIATIONS:
            if total == rule_total and upcard in upcards and true_count <= ceiling:
                return PlayerDecision.HIT
        return None

    @staticmethod
    def decide_counting(observation) -> PlayerDecision:
        if observation is None:
            return PlayerDecision.HIT
        deviation = Player.counting_deviation(observation)
        if deviation is not None:
            return deviation
        return Player.decide_basic_strategy(observation)

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
        strategy_deciders = {
            PlayerType.BASIC: Player.decide_basic_strategy,
            PlayerType.COUNTING: Player.decide_counting,
        }
        if player_type in strategy_deciders:
            return strategy_deciders[player_type](decision_info.observation)
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

    def count_bet_units(self) -> int:
        return max(1, min(BET_UNIT_CAP, int(self.true_count)))

    def get_bet_units(self) -> int:
        if self.player_type == PlayerType.COUNTING:
            return self.count_bet_units()
        return self.bet_units

    def get_bet_amount(self) -> int:
        return self.base_bet * self.get_bet_units()

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
        if isinstance(self.player_type, PlayerType):
            return self.player_type.name
        return "UNKNOWN"
