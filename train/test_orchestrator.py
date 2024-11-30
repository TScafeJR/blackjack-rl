import json
import os
import tempfile
import unittest

from .config import TrainingConfig
from .environment import BET_UNITS
from .orchestrator import TrainingRun


class TestTrainingRunSync(unittest.TestCase):
    def test_execute_smoke_run(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            config = TrainingConfig(
                agents="dqn=1,mc=1,random=1,noob=1",
                workers=0,
                tables=1,
                hands=300,
                seed=7,
                sync_interval=10,
                train_interval=4,
                buffer_size=1000,
                batch_size=16,
                run_dir=temp_dir,
            )

            run_path = TrainingRun(config).execute()

            self.assertTrue(os.path.exists(os.path.join(run_path, "config.json")))
            self.assertTrue(os.path.exists(os.path.join(run_path, "metrics.jsonl")))
            self.assertTrue(os.path.exists(os.path.join(run_path, "report.txt")))
            self.assertTrue(os.path.exists(os.path.join(run_path, "weights_dqn.json")))
            self.assertTrue(os.path.exists(os.path.join(run_path, "weights_mc.json")))

            with open(
                os.path.join(run_path, "metrics.jsonl"), "r", encoding="utf-8"
            ) as metrics_file:
                record_count = sum(1 for line in metrics_file if line.strip())
            self.assertGreaterEqual(record_count, 300)


class TestTrainingRunCounting(unittest.TestCase):
    def test_ramp_run_saves_both_networks_and_count_metrics(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            config = TrainingConfig(
                agents="dqn-ramp=1,mc-count=1,counting=1,basic=1",
                workers=0,
                tables=1,
                hands=300,
                seed=7,
                sync_interval=10,
                train_interval=4,
                buffer_size=1000,
                batch_size=16,
                dealer_rule="s17",
                reward_scale="initial_bet",
                run_dir=temp_dir,
            )

            run_path = TrainingRun(config).execute()

            for name in ["dqn-ramp", "dqn-ramp-bet", "mc-count"]:
                self.assertTrue(
                    os.path.exists(os.path.join(run_path, f"weights_{name}.json")),
                    name,
                )
            self.assertFalse(
                os.path.exists(os.path.join(run_path, "weights_mc-count-bet.json"))
            )

            with open(
                os.path.join(run_path, "config.json"), "r", encoding="utf-8"
            ) as config_file:
                saved = json.load(config_file)
            self.assertEqual(saved["dealer_rule"], "s17")
            self.assertEqual(saved["reward_scale"], "initial_bet")

            with open(
                os.path.join(run_path, "metrics.jsonl"), "r", encoding="utf-8"
            ) as metrics_file:
                records = [json.loads(line) for line in metrics_file if line.strip()]
            self.assertTrue(all("true_count" in record for record in records))
            ramp_bets = {
                record["bet_units"] for record in records if record["kind"] == "dqn-ramp"
            }
            self.assertTrue(ramp_bets.issubset(set(BET_UNITS)))


class TestTrainingRunParallel(unittest.TestCase):
    def test_execute_parallel_smoke_run(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            config = TrainingConfig(
                agents="dqn=2,random=2",
                workers=2,
                tables=2,
                hands=400,
                seed=7,
                sync_interval=5,
                train_interval=4,
                buffer_size=1000,
                batch_size=16,
                run_dir=temp_dir,
            )

            run_path = TrainingRun(config).execute()

            self.assertTrue(os.path.exists(os.path.join(run_path, "metrics.jsonl")))
            self.assertTrue(os.path.exists(os.path.join(run_path, "weights_dqn.json")))
            with open(
                os.path.join(run_path, "metrics.jsonl"), "r", encoding="utf-8"
            ) as metrics_file:
                record_count = sum(1 for line in metrics_file if line.strip())
            self.assertGreaterEqual(record_count, 400)


if __name__ == "__main__":
    unittest.main()
