from typing import List
from .card import Card
from .hand import Hand


class BasePlayer:
    def __init__(self):
        self.hand = Hand()

    def receive_card(self, new_card: Card) -> None:
        self.hand.add_card(new_card)

    def return_cards(self) -> List[Card]:
        return self.hand.return_cards()

    def see_hand(self) -> List[Card]:
        return self.hand.show_cards()

    def get_hand_values(self) -> List[int]:
        return self.hand.get_values()

    def has_blackjack(self) -> bool:
        return self.hand.includes_blackjack()

    def has_bust_hand(self) -> bool:
        return self.hand.is_bust()

    def get_hand_value(self) -> int:
        return self.hand.get_high_value()
