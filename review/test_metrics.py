import unittest

from .metrics import (MetricsCollector, format_count_table, format_summary,
                      summarize_by_count, summarize_hands, tail_records)


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
        **{
            key: kwargs[key]
            for key in ["base_bet", "bet_units", "true_count"]
            if key in kwargs
        },
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


class TestUnitMetrics(unittest.TestCase):
    def test_units_fall_back_to_a_flat_ten_bet(self):
        summaries = summarize_hands([build_record("dqn", "PLAYER_WIN", 1.0)])
        self.assertAlmostEqual(summaries["dqn"]["avg_units"], 1.0, places=10)
        self.assertAlmostEqual(summaries["dqn"]["avg_bet_units"], 1.0, places=10)

    def test_units_use_the_recorded_base_bet(self):
        records = [
            build_record("ramp", "PLAYER_WIN", 1.0, bet=50, base_bet=10, bet_units=5),
            build_record("ramp", "BUST", -1.0, bet=10, base_bet=10, bet_units=1),
        ]

        summaries = summarize_hands(records)

        self.assertAlmostEqual(summaries["ramp"]["net_units"], 4.0, places=10)
        self.assertAlmostEqual(summaries["ramp"]["avg_units"], 2.0, places=10)
        self.assertAlmostEqual(summaries["ramp"]["avg_bet_units"], 3.0, places=10)

    def test_ev_per_unit_ignores_bet_size(self):
        records = [
            build_record("ramp", "PLAYER_WIN", 1.0, bet=50, base_bet=10, bet_units=5),
            build_record("flat", "PLAYER_WIN", 1.0, bet=10, base_bet=10, bet_units=1),
        ]

        summaries = summarize_hands(records)

        self.assertAlmostEqual(summaries["ramp"]["avg_reward"], 1.0, places=10)
        self.assertAlmostEqual(summaries["flat"]["avg_reward"], 1.0, places=10)
        self.assertAlmostEqual(summaries["ramp"]["avg_units"], 5.0, places=10)
        self.assertAlmostEqual(summaries["flat"]["avg_units"], 1.0, places=10)

    def test_doubled_hands_wager_more_than_their_bet_units(self):
        records = [
            build_record("ramp", "PLAYER_WIN", 1.0, bet=40, base_bet=10, bet_units=2)
        ]

        summaries = summarize_hands(records)

        self.assertAlmostEqual(summaries["ramp"]["avg_bet_units"], 2.0, places=10)
        self.assertAlmostEqual(summaries["ramp"]["avg_units"], 4.0, places=10)

    def test_format_summary_reports_units(self):
        text = format_summary(summarize_hands([build_record("dqn", "PUSH", 0.0)]))
        self.assertIn("units/hand", text)
        self.assertIn("ev/unit", text)
        self.assertIn("avg bet", text)


class TestTailRecords(unittest.TestCase):
    def setUp(self):
        self.records = [
            build_record("dqn", "PLAYER_WIN", 1.0, player_id=f"dqn-{index}")
            for index in range(10)
        ] + [
            build_record("random", "BUST", -1.0, player_id=f"random-{index}")
            for index in range(4)
        ]

    def test_full_share_returns_everything(self):
        self.assertEqual(len(tail_records(self.records, 1.0)), 14)

    def test_tail_is_taken_per_kind(self):
        kept = tail_records(self.records, 0.5)
        by_kind = {}
        for record in kept:
            by_kind[record["kind"]] = by_kind.get(record["kind"], 0) + 1
        self.assertEqual(by_kind, {"dqn": 5, "random": 2})

    def test_tail_keeps_the_most_recent_hands(self):
        kept = tail_records(self.records, 0.2)
        dqn_ids = [r["player_id"] for r in kept if r["kind"] == "dqn"]
        self.assertEqual(dqn_ids, ["dqn-8", "dqn-9"])

    def test_tiny_share_keeps_at_least_one_hand(self):
        kept = tail_records(self.records, 0.001)
        self.assertEqual(len(kept), 2)


class TestCountTable(unittest.TestCase):
    def test_reports_nothing_without_counts(self):
        text = format_count_table([build_record("dqn", "PUSH", 0.0)])
        self.assertIn("no true count recorded", text)

    def test_buckets_by_rounded_true_count(self):
        records = [
            build_record("ramp", "PLAYER_WIN", 1.0, bet=50, bet_units=5, true_count=3.4),
            build_record("ramp", "BUST", -1.0, bet=10, bet_units=1, true_count=-2.6),
        ]

        by_kind = summarize_by_count(records)

        self.assertEqual(by_kind["ramp"][3]["avg_bet_units"], 5.0)
        self.assertEqual(by_kind["ramp"][-3]["avg_bet_units"], 1.0)

    def test_extreme_counts_clamp_into_end_buckets(self):
        records = [
            build_record("ramp", "PUSH", 0.0, true_count=42.0),
            build_record("ramp", "PUSH", 0.0, true_count=-42.0),
        ]

        buckets = summarize_by_count(records)["ramp"]

        self.assertEqual(sorted(buckets), [-5, 5])

    def test_ev_per_unit_divides_by_wager(self):
        records = [
            build_record("ramp", "PLAYER_WIN", 1.0, bet=50, bet_units=5, true_count=2.0)
        ]

        bucket = summarize_by_count(records)["ramp"][2]

        self.assertAlmostEqual(bucket["avg_units"], 5.0, places=10)
        self.assertAlmostEqual(bucket["ev_per_unit"], 1.0, places=10)

    def test_table_lists_both_metrics(self):
        text = format_count_table(
            [build_record("ramp", "PUSH", 0.0, true_count=1.0, bet_units=2)]
        )
        self.assertIn("avg bet", text)
        self.assertIn("units/hand", text)
        self.assertIn("ramp", text)


if __name__ == "__main__":
    unittest.main()
