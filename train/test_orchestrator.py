import os
import tempfile
import unittest

from .config import TrainingConfig
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
