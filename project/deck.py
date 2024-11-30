from __future__ import annotations

import random
from typing import List

from .card import Card


class Deck:
    def __init__(self):
        self.cards = self.add_cards()
        self.discarded_cards: List[Card] = []
        self.shuffle_epoch = 0
        self.shuffle_cards()

    @staticmethod
    def add_cards() -> List[Card]:
        suits = ["Spades", "Hearts", "Clubs", "Diamonds"]
        displays = ["A", "2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K"]
        cards = []

        for suit in suits:
            for val in displays:
                new_card = Card(val, suit)
                cards.append(new_card)

        return cards

    def shuffle_cards(self):
        random.shuffle(self.cards)

    def draw_card(self):
        if len(self.cards) == 0:
            self.shuffle_empty_deck()
        drawn_card = self.cards.pop()
        self.discarded_cards.append(drawn_card)
        return drawn_card

    def combine_deck_and_shuffle(self, incoming_deck: Deck):
        self.cards.extend(incoming_deck.cards)
        self.shuffle_cards()

    def shuffle_empty_deck(self):
        self.cards = self.discarded_cards
        self.discarded_cards = []
        self.shuffle_epoch += 1
        self.shuffle_cards()
