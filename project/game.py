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


class DecisionInfo:
    def __init__(self, **kwargs):
        self.min_bet = kwargs.get("min_bet", 10)
        self.max_bet = kwargs.get("max_bet", 10)
        self.stage = kwargs.get("stage", TurnStage.SUBMITTING_BET)
        self.player_bet = 0
        self.action_reward = 0

    def set_player_bet(self, bet: int):
        self.player_bet = bet
