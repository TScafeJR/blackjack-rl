import unittest

from .config import TrainingConfig, config_from_args, parse_agents


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


if __name__ == "__main__":
    unittest.main()
