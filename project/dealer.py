from .base_player import BasePlayer
from .deck import Deck
from .card import Card


class Dealer(BasePlayer):
    def __init__(self):
        super().__init__()

    @staticmethod
    def deal_player_initial_cards(player: BasePlayer, table_deck: Deck) -> None:
        started_with_one = len(table_deck.cards) == 1
        started_with_two = len(table_deck.cards) == 2

        if len(table_deck.cards) == 0:
            table_deck.shuffle_empty_deck()

        if started_with_one:
            card1 = table_deck.draw_card()
            player.receive_card(card1)
            table_deck.shuffle_empty_deck()
            card2 = table_deck.draw_card()
            player.receive_card(card2)
            return

        card1 = table_deck.draw_card()
        card2 = table_deck.draw_card()
        player.receive_card(card1)
        player.receive_card(card2)

        if started_with_two:
            table_deck.shuffle_empty_deck()

    def deal_self_cards(self, table_deck: Deck) -> None:
        self.deal_player_initial_cards(self, table_deck)

    def preview_card(self) -> Card:
        return self.cards[0]

    @staticmethod
    def deal_player_card(player: BasePlayer, deck: Deck):
        card = deck.draw_card()
        player.receive_card(card)

        if len(deck.cards) == 0:
            deck.shuffle_empty_deck()

    def deal_self_card(self, deck: Deck) -> None:
        self.deal_player_card(self, deck)
