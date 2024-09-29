import unittest

from .metrics import MetricsCollector, format_summary, summarize_hands


def build_record(kind, result, reward, **kwargs):
    return {
        "kind": kind,
        "result": result,
        "reward": reward,
        "bet": kwargs.get("bet", 10),
        "money_after": kwargs.get("money_after", 1000),
        "rebuys": kwargs.get("rebuys", 0),
        "player_id": kwargs.get("player_id", "p1"),
        "epsilon": kwargs.get("epsilon", 0.0),
    }


class TestMetricsCollector(unittest.TestCase):
    def test_record_hands_accumulates(self):
        collector = MetricsCollector()
        collector.record_hands([build_record("dqn", "PLAYER_WIN", 1.0)])
        collector.record_hands([build_record("dqn", "BUST", -1.0)])
        self.assertEqual(len(collector.hand_records), 2)


class TestSummarizeHands(unittest.TestCase):
    def setUp(self):
        self.records = [
            build_record("dqn", "PLAYER_WIN", 1.0),
            build_record("dqn", "DEALER_BUST", 1.0),
            build_record("dqn", "PLAYER_BLACKJACK", 1.5),
            build_record("dqn", "PUSH", 0.0),
            build_record("dqn", "BUST", -1.0),
            build_record("dqn", "DEALER_WIN", -1.0, rebuys=2),
            build_record("random", "BUST", -2.0, bet=20),
        ]

    def test_summarize_counts(self):
        summaries = summarize_hands(self.records)
        dqn = summaries["dqn"]

        self.assertEqual(dqn["hands"], 6)
        self.assertEqual(dqn["wins"], 2)
        self.assertEqual(dqn["blackjacks"], 1)
        self.assertEqual(dqn["pushes"], 1)
        self.assertEqual(dqn["busts"], 1)
        self.assertEqual(dqn["losses"], 2)
        self.assertEqual(dqn["rebuys"], 2)
        self.assertAlmostEqual(dqn["win_rate"], 0.5, places=10)
        self.assertAlmostEqual(dqn["avg_reward"], 1.5 / 6, places=10)
        self.assertAlmostEqual(dqn["net_profit"], 15.0, places=10)

    def test_summarize_uses_bet_for_profit(self):
        summaries = summarize_hands(self.records)
        self.assertAlmostEqual(summaries["random"]["net_profit"], -40.0, places=10)

    def test_format_summary_contains_kinds(self):
        text = format_summary(summarize_hands(self.records))
        self.assertIn("dqn", text)
        self.assertIn("random", text)
        self.assertIn("win %", text)


if __name__ == "__main__":
    unittest.main()
