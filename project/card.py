from typing import List


class Card:
    special_cards = ["J", "Q", "K"]

    def __init__(self, display: str, suit: str):
        self.display = display
        self.suit = suit

    def get_display(self) -> str:
        return self.display

    def get_value(self) -> List[int]:
        if self.display == "A":
            return 0
        if self.display in self.special_cards:
            return 10
        else:
            return int(self.display)

    def to_string(self):
        return f"{self.display} of {self.suit}"
