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
