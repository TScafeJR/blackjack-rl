from typing import List
from .card import Card
from .game import Game


class Hand:
    def __init__(self):
        self.cards = []

    def add_card(self, card: Card):
        self.cards.append(card)

    def return_cards(self) -> List[Card]:
        c = self.cards
        self.cards = []
        return c

    def get_values(self) -> List[int]:
        possible_totals = [0]

        for card in self.cards:
            card_values = card.get_value()
            new_totals = []

            for value in card_values:
                for total in possible_totals:
                    new_totals.append(total + value)

            possible_totals = new_totals

        # Remove duplicates and sort the possible totals
        possible_totals = list(set(possible_totals))
        possible_totals.sort()

        return possible_totals

    def get_high_value(self) -> int:
        vals = self.get_values()
        highest_valid_hand_val = vals[0]

        for val in vals:
            if val > highest_valid_hand_val and val <= Game.BLACKJACK_SCORE:
                highest_valid_hand_val = val

        return highest_valid_hand_val

    def includes_blackjack(self) -> bool:
        vals = self.get_values()
        for val in vals:
            if val == Game.BLACKJACK_SCORE:
                return True

        return False

    def is_bust(self) -> bool:
        return self.get_high_value() > Game.BLACKJACK_SCORE

    def show_cards(self) -> List[Card]:
        return list(map(lambda x: x.to_string(), self.cards))

    def preview_card(self) -> Card:
        return self.cards[0]
