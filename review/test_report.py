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


if __name__ == "__main__":
    unittest.main()
