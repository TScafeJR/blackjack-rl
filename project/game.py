from enum import Enum


class Game:
    BLACKJACK_SCORE = 21
    DEALER_HIT_CEIL = 16


class HandResult(Enum):
    UNSPECIFIED = "unspecified"
    PUSH = "push"
    BUST = "player_bust"
    DEALER_WIN = "dealer_win"
    PLAYER_WIN = "player_win"
    PLAYER_BLACKJACK = "player_blackjack"
    DEALER_BUST = "dealer_bust"


class TurnStage(Enum):
    UNSPECIFIED = "unspecified"
    SUBMITTING_BET = "submitting_bet"
    PLAYING = "playing"
    WAITING_RESULT = "waiting_result"


class Observation:
    def __init__(self, **kwargs):
        self.player_total = kwargs.get("player_total", 0)
        self.is_soft = kwargs.get("is_soft", False)
        self.dealer_upcard_value = kwargs.get("dealer_upcard_value", 0)
        self.can_double = kwargs.get("can_double", False)
        self.money = kwargs.get("money", 0)


class PendingTurn:
    def __init__(self, **kwargs):
        self.player_id = kwargs.get("player_id", "")
        self.observation = kwargs.get("observation", None)
        self.legal_actions = kwargs.get("legal_actions", [])


class HandOutcome:
    def __init__(self, **kwargs):
        self.result = kwargs.get("result", HandResult.UNSPECIFIED)
        self.reward = kwargs.get("reward", 0)
        self.bet = kwargs.get("bet", 0)
        self.money_after = kwargs.get("money_after", 0)


class DecisionInfo:
    def __init__(self, **kwargs):
        self.min_bet = kwargs.get("min_bet", 10)
        self.max_bet = kwargs.get("max_bet", 10)
        self.stage = kwargs.get("stage", TurnStage.PLAYING)
        self.observation = kwargs.get("observation", None)
        self.legal_actions = kwargs.get("legal_actions", [])
