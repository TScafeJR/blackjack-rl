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
                    new_total = total + value
                    if new_total <= 21:
                        new_totals.append(new_total)

            possible_totals = new_totals or possible_totals

        if possible_totals:
            return list(set(possible_totals))
        else:
            return [min(sum(card.get_value()) for card in self.cards)]

    def get_high_value(self) -> int:
        vals = self.get_values()
        highest_valid_hand_val = vals[0]
        for val in vals:
            if val > highest_valid_hand_val and highest_valid_hand_val <= Game.BLACKJACK_SCORE:
                highest_valid_hand_val = val
        return highest_valid_hand_val

    def includes_blackjack(self) -> bool:
        vals = self.get_values()
        for val in vals:
            if val == Game.BLACKJACK_SCORE:
                return True

        return False

    def is_bust(self) -> bool:
        vals = self.get_values()
        bust_vals = 0
        for val in vals:
            if val > Game.BLACKJACK_SCORE:
                bust_vals += 1

        return bust_vals == len(self.cards)

    def show_cards(self) -> List[Card]:
        return list(map(lambda x: x.to_string(), self.cards))
