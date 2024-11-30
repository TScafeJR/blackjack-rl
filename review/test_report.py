import tempfile
import unittest

from .report import build_comparison, build_report
from .test_plots import build_synthetic_run


class TestReport(unittest.TestCase):
    def test_build_report(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            store = build_synthetic_run(temp_dir)

            report = build_report(store.run_path)

            self.assertIn("dqn", report)
            self.assertIn("random", report)
            self.assertIn("agents: dqn=1,random=1", report)

    def test_build_report_missing_run(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            with self.assertRaises(Exception):
                build_report(temp_dir + "/missing")

    def test_build_comparison(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            store = build_synthetic_run(temp_dir)

            comparison = build_comparison([store.run_path, store.run_path])

            self.assertIn("dqn", comparison)
            self.assertIn("test-run", comparison)


class TestReportScoping(unittest.TestCase):
    def test_tail_narrows_the_hand_count(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            store = build_synthetic_run(temp_dir)

            full = build_report(store.run_path)
            tail = build_report(store.run_path, tail=0.25)

            self.assertNotIn("scope:", full)
            self.assertIn("scope: last 25%", tail)
            self.assertNotEqual(full, tail)

    def test_counts_section_is_opt_in(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            store = build_synthetic_run(temp_dir)

            self.assertNotIn("true count", build_report(store.run_path, counts=False))
            self.assertIn("true count", build_report(store.run_path, counts=True))

    def test_counts_section_tabulates_recorded_counts(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            store = build_synthetic_run(temp_dir)
            store.append_hands(
                [
                    {
                        "kind": "ramp",
                        "player_id": "ramp-1",
                        "result": "PLAYER_WIN",
                        "reward": 1.0,
                        "bet": 50,
                        "money_after": 1000,
                        "epsilon": 0.0,
                        "rebuys": 0,
                        "true_count": 4.2,
                        "bet_units": 5,
                        "base_bet": 10,
                    }
                ]
            )

            report = build_report(store.run_path, counts=True)

            self.assertIn("bet size and profit by true count", report)
            self.assertIn("ramp", report)


if __name__ == "__main__":
    unittest.main()
