import unittest

from project import DealerRule

from .config import (LEARNING_SPECS, TrainingConfig, config_from_args,
                     parse_agents)


class TestParseAgents(unittest.TestCase):
    def test_parse_agents(self):
        self.assertEqual(
            parse_agents("dqn=2, mc=1, random=1"),
            [("dqn", 2), ("mc", 1), ("random", 1)],
        )

    def test_parse_agents_defaults_to_one(self):
        self.assertEqual(parse_agents("dqn"), [("dqn", 1)])

    def test_parse_agents_unknown_kind(self):
        with self.assertRaises(Exception):
            parse_agents("alphazero=1")


class TestTrainingConfig(unittest.TestCase):
    def test_validate_too_many_players(self):
        with self.assertRaises(Exception):
            TrainingConfig(agents="dqn=5", tables=1)

    def test_validate_too_few_players(self):
        with self.assertRaises(Exception):
            TrainingConfig(agents="dqn=1", tables=2)

    def test_validate_workers_exceed_tables(self):
        with self.assertRaises(Exception):
            TrainingConfig(agents="dqn=2", tables=1, workers=2)

    def test_decay_hands_for_kind_scales_by_seats(self):
        config = TrainingConfig(agents="dqn=2,random=2", tables=1, hands=1000)
        self.assertEqual(config.decay_hands_for_kind("dqn"), 300)

    def test_decay_hands_for_kind_explicit_override(self):
        config = TrainingConfig(
            agents="dqn=2,random=2", tables=1, hands=1000, epsilon_decay_hands=42
        )
        self.assertEqual(config.decay_hands_for_kind("dqn"), 42)

    def test_config_from_args(self):
        config = config_from_args(
            ["--agents", "dqn=2,mc=2", "--tables", "1", "--hands", "500"]
        )
        self.assertEqual(config.total_seats(), 4)
        self.assertEqual(config.hands, 500)
        self.assertEqual(config.learning_kinds_in_use(), ["dqn", "mc"])


class TestLearningSpecs(unittest.TestCase):
    def test_plain_kinds_are_blind_and_flat_betting(self):
        for kind in ["dqn", "mc"]:
            self.assertFalse(LEARNING_SPECS[kind].uses_count)
            self.assertFalse(LEARNING_SPECS[kind].learns_bet)

    def test_count_kinds_see_the_count_without_betting(self):
        for kind in ["dqn-count", "mc-count"]:
            self.assertTrue(LEARNING_SPECS[kind].uses_count)
            self.assertFalse(LEARNING_SPECS[kind].learns_bet)

    def test_ramp_kinds_see_the_count_and_bet(self):
        for kind in ["dqn-ramp", "mc-ramp"]:
            self.assertTrue(LEARNING_SPECS[kind].uses_count)
            self.assertTrue(LEARNING_SPECS[kind].learns_bet)

    def test_algorithm_is_carried_through(self):
        self.assertEqual(LEARNING_SPECS["mc-ramp"].algorithm, "mc")
        self.assertEqual(LEARNING_SPECS["dqn-count"].algorithm, "dqn")

    def test_variant_kinds_parse_as_agents(self):
        self.assertEqual(
            parse_agents("dqn-ramp=2,counting=1"), [("dqn-ramp", 2), ("counting", 1)]
        )


class TestDealerRuleConfig(unittest.TestCase):
    def test_defaults_to_the_house_soft_rule(self):
        config = TrainingConfig(agents="dqn=1", tables=1)
        self.assertEqual(config.dealer_rule_value(), DealerRule.SOFT_ANY)

    def test_standard_rules_resolve(self):
        for name, expected in [
            ("h17", DealerRule.HIT_SOFT_17),
            ("s17", DealerRule.STAND_SOFT_17),
        ]:
            config = TrainingConfig(agents="dqn=1", tables=1, dealer_rule=name)
            self.assertEqual(config.dealer_rule_value(), expected)

    def test_unknown_rule_is_rejected(self):
        with self.assertRaises(Exception):
            TrainingConfig(agents="dqn=1", tables=1, dealer_rule="stands_on_12")

    def test_rule_is_saved_in_the_run_config(self):
        config = TrainingConfig(agents="dqn=1", tables=1, dealer_rule="s17")
        self.assertEqual(config.to_dict()["dealer_rule"], "s17")


if __name__ == "__main__":
    unittest.main()
