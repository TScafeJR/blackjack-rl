import unittest
from unittest.mock import MagicMock
from .hand import Hand
from .card import Card


class TestHand(unittest.TestCase):
    def setUp(self):
        # Create a Hand instance for testing
        self.hand = Hand()

    def test_add_card(self):
        card1 = Card("A", "Spades")
        card2 = Card("5", "Hearts")

        self.hand.add_card(card1)
        self.assertEqual(len(self.hand.cards), 1)

        self.hand.add_card(card2)
        self.assertEqual(len(self.hand.cards), 2)

    def test_return_cards(self):
        card1 = Card("A", "Spades")
        card2 = Card("5", "Hearts")

        self.hand.add_card(card1)
        self.hand.add_card(card2)

        returned_cards = self.hand.return_cards()
        self.assertEqual(len(returned_cards), 2)
        self.assertEqual(len(self.hand.cards), 0)

    def test_get_values(self):
        card1 = Card("A", "Spades")
        card2 = Card("5", "Hearts")
        card3 = Card("K", "Diamonds")

        self.hand.add_card(card1)
        self.hand.add_card(card2)
        self.hand.add_card(card3)

        # Mock the get_value method of Card to return specific values
        card1.get_value = MagicMock(return_value=[1, 11])
        card2.get_value = MagicMock(return_value=[5])
        card3.get_value = MagicMock(return_value=[10])

        # Test get_values method
        values = self.hand.get_values()
        self.assertIn(16, values)  # A (1 or 11) + 5 + K (10)
        self.assertIn(26, values)  # A (1 or 11) + 5 + K (10

    def test_get_high_value(self):
        card1 = Card("A", "Spades")
        card2 = Card("5", "Hearts")
        card3 = Card("K", "Diamonds")

        self.hand.add_card(card1)
        self.hand.add_card(card2)
        self.hand.add_card(card3)

        # Mock the get_value method of Card to return specific values
        card1.get_value = MagicMock(return_value=[1, 11])
        card2.get_value = MagicMock(return_value=[5])
        card3.get_value = MagicMock(return_value=[10])

        # Test get_high_value method
        high_value = self.hand.get_high_value()
        self.assertEqual(high_value, 16)  # Highest valid hand value <= 21

    def test_get_high_value_bust(self):
        card1 = Card("7", "Spades")
        card2 = Card("5", "Hearts")
        card3 = Card("K", "Diamonds")

        self.hand.add_card(card1)
        self.hand.add_card(card2)
        self.hand.add_card(card3)

        # Mock the get_value method of Card to return specific values
        card1.get_value = MagicMock(return_value=[7])
        card2.get_value = MagicMock(return_value=[5])
        card3.get_value = MagicMock(return_value=[10])

        # Test get_high_value method
        high_value = self.hand.get_high_value()
        self.assertEqual(high_value, 22)  # Highest valid hand value <= 21

    def test_includes_blackjack(self):
        card1 = Card("A", "Spades")
        card2 = Card("K", "Hearts")

        self.hand.add_card(card1)
        self.hand.add_card(card2)

        # Mock the get_value method of Card to return specific values
        card1.get_value = MagicMock(return_value=[1, 11])
        card2.get_value = MagicMock(return_value=[10])

        # Test includes_blackjack method
        self.assertTrue(self.hand.includes_blackjack())

    def test_is_bust(self):
        card1 = Card("10", "Spades")
        card2 = Card("J", "Hearts")
        card3 = Card("3", "Diamonds")

        self.hand.add_card(card1)
        self.hand.add_card(card2)

        # Mock the get_value method of Card to return specific values
        card1.get_value = MagicMock(return_value=[10])
        card2.get_value = MagicMock(return_value=[10])
        self.assertFalse(self.hand.is_bust())

        self.hand.add_card(card3)
        card3.get_value = MagicMock(return_value=[3])

        self.assertTrue(self.hand.is_bust())

    def test_show_cards(self):
        card1 = Card("Q", "Spades")
        card2 = Card("5", "Hearts")

        self.hand.add_card(card1)
        self.hand.add_card(card2)

        # Test show_cards method
        cards_str = self.hand.show_cards()
        self.assertEqual(cards_str, ["Q of Spades", "5 of Hearts"])


if __name__ == "__main__":
    unittest.main()
