import random
import unittest

from .deck import Deck


class TestDeck(unittest.TestCase):
    def setUp(self):
        random.seed(13)
        self.deck = Deck()

    def test_init(self):
        self.assertEqual(len(self.deck.cards), 52)
        self.assertEqual(len(self.deck.discarded_cards), 0)

        card_names = [card.to_string() for card in self.deck.cards]
        self.assertEqual(len(set(card_names)), 52)

    def test_init_shuffles(self):
        unshuffled_names = [card.to_string() for card in Deck.add_cards()]
        shuffled_names = [card.to_string() for card in self.deck.cards]
        self.assertNotEqual(shuffled_names, unshuffled_names)

    def test_init_deterministic_under_seed(self):
        random.seed(13)
        other_deck = Deck()
        self.assertEqual(
            [card.to_string() for card in self.deck.cards],
            [card.to_string() for card in other_deck.cards],
        )

    def test_draw_card(self):
        top_card = self.deck.cards[-1]
        drawn_card = self.deck.draw_card()

        self.assertIs(drawn_card, top_card)
        self.assertEqual(len(self.deck.cards), 51)
        self.assertEqual(len(self.deck.discarded_cards), 1)

    def test_draw_card_reshuffles_empty_deck(self):
        for _ in range(52):
            self.deck.draw_card()
        self.assertEqual(len(self.deck.cards), 0)

        drawn_card = self.deck.draw_card()

        self.assertIsNotNone(drawn_card)
        self.assertEqual(len(self.deck.cards), 51)
        self.assertEqual(len(self.deck.discarded_cards), 1)

    def test_combine_deck_and_shuffle(self):
        self.deck.combine_deck_and_shuffle(Deck())
        self.assertEqual(len(self.deck.cards), 104)

    def test_shuffle_empty_deck_increments_epoch(self):
        self.assertEqual(self.deck.shuffle_epoch, 0)
        for _ in range(53):
            self.deck.draw_card()
        self.assertEqual(self.deck.shuffle_epoch, 1)


if __name__ == "__main__":
    unittest.main()
