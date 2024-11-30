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
            return [1, 11]
        if self.display in self.special_cards:
            return [10]

        return [int(self.display)]

    def hi_lo_value(self) -> int:
        values = self.get_value()
        if len(values) > 1 or values[0] == 10:
            return -1
        if values[0] <= 6:
            return 1
        return 0

    def to_string(self) -> str:
        return f"{self.display} of {self.suit}"
