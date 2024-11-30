import unittest

from .card import Card
from .dealer import Dealer
from .game import DealerRule


def build_dealer(rule, displays):
    dealer = Dealer(dealer_rule=rule)
    for display in displays:
        dealer.receive_card(Card(display, "Spades"))
    return dealer


class TestDealerRules(unittest.TestCase):
    def test_soft_any_hits_every_soft_hand(self):
        for displays in [["A", "6"], ["A", "7"], ["A", "9"], ["A", "10"]]:
            self.assertTrue(build_dealer(DealerRule.SOFT_ANY, displays).can_hit())

    def test_hit_soft_17_hits_soft_17_only(self):
        self.assertTrue(build_dealer(DealerRule.HIT_SOFT_17, ["A", "6"]).can_hit())
        self.assertFalse(build_dealer(DealerRule.HIT_SOFT_17, ["A", "7"]).can_hit())
        self.assertFalse(build_dealer(DealerRule.HIT_SOFT_17, ["A", "9"]).can_hit())

    def test_stand_soft_17_stands_on_every_seventeen(self):
        self.assertFalse(build_dealer(DealerRule.STAND_SOFT_17, ["A", "6"]).can_hit())
        self.assertFalse(build_dealer(DealerRule.STAND_SOFT_17, ["10", "7"]).can_hit())

    def test_standard_rules_hit_hard_sixteen(self):
        for rule in [DealerRule.HIT_SOFT_17, DealerRule.STAND_SOFT_17]:
            self.assertTrue(build_dealer(rule, ["10", "6"]).can_hit())

    def test_standard_rules_stand_on_hard_eighteen(self):
        for rule in [DealerRule.HIT_SOFT_17, DealerRule.STAND_SOFT_17]:
            self.assertFalse(build_dealer(rule, ["10", "8"]).can_hit())

    def test_defaults_to_soft_any(self):
        self.assertEqual(Dealer().dealer_rule, DealerRule.SOFT_ANY)

    def test_bust_hand_never_hits(self):
        for rule in DealerRule:
            self.assertFalse(build_dealer(rule, ["10", "9", "5"]).can_hit())


if __name__ == "__main__":
    unittest.main()
