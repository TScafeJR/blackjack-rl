import unittest

from .card import Card


class TestCard(unittest.TestCase):
    def setUp(self):
        # Create sample cards for testing
        self.card1 = Card("A", "Spades")
        self.card2 = Card("5", "Hearts")
        self.card3 = Card("K", "Diamonds")

    def test_get_display(self):
        self.assertEqual(self.card1.get_display(), "A")
        self.assertEqual(self.card2.get_display(), "5")
        self.assertEqual(self.card3.get_display(), "K")

    def test_get_value(self):
        self.assertEqual(self.card1.get_value(), [1, 11])  # Ace can be 1 or 11
        self.assertEqual(self.card2.get_value(), [5])  # Number cards return their value
        self.assertEqual(self.card3.get_value(), [10])  # Face cards (K) return 10

    def test_to_string(self):
        self.assertEqual(self.card1.to_string(), "A of Spades")
        self.assertEqual(self.card2.to_string(), "5 of Hearts")
        self.assertEqual(self.card3.to_string(), "K of Diamonds")


if __name__ == "__main__":
    unittest.main()
