from typing import List
from .card import Card


class BasePlayer:
    def __init__(self):
        self.cards = []

    def receive_card(self, new_card: Card) -> None:
        self.cards.append(new_card)

    def return_cards(self) -> List[int]:
        self.cards = []

    def see_hand(self) -> List[Card]:
        return list(map(lambda x: x.to_string(), self.cards))

    def get_hand_value(self) -> int:
        non_ace_cards = []
        num_aces = 0

        for card in self.cards:
            if card.get_display() == "A":
                num_aces += 1
            else:
                non_ace_cards.append(card)

        non_ace_total = 0
        for card in non_ace_cards:
            non_ace_total += card.get_value()

        if num_aces > 0:
            if non_ace_total + 11 + (num_aces - 1) * 1 <= 21:
                ace_value = 11 + (num_aces - 1) * 1
            else:
                ace_value = num_aces * 1
        else:
            ace_value = 0

        return non_ace_total + ace_value
