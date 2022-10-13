from __future__ import annotations

from .card import Card
import random


class Deck:
    def __init__(self):
        self.cards = self.add_cards()
        self.discarded_cards = []

    @staticmethod
    def add_cards():
        suits = {"Spades", "Hearts", "Clubs", "Diamonds"}
        displays = {"A", "2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K"}
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
        discarded_card = self.cards.pop()
        self.discarded_cards.append(discarded_card)
        return discarded_card

    def combine_deck_and_shuffle(self, incoming_deck: Deck):
        incoming_cards = incoming_deck.cards
        self.cards.extend(incoming_cards)
        self.shuffle_cards()

    def shuffle_empty_deck(self):
        self.cards = self.discarded_cards
        self.discarded_cards = []
        self.shuffle_cards()

    def get_cards(self):
        return self.cards
