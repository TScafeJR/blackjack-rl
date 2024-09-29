import os
import random
import tempfile
import unittest

from .plots import render_all, rolling_mean
from .run_store import RunStore


def build_synthetic_run(temp_dir):
    store = RunStore(base_dir=temp_dir, run_name="test-run")
    rng = random.Random(3)
    results = ["PLAYER_WIN", "DEALER_WIN", "BUST", "PUSH", "PLAYER_BLACKJACK"]
    hand_records = []
    for index in range(200):
        kind = "dqn" if index % 2 == 0 else "random"
        result = rng.choice(results)
        hand_records.append(
            {
                "kind": kind,
                "player_id": f"{kind}-player",
                "result": result,
                "reward": 1.0 if result == "PLAYER_WIN" else -1.0,
                "bet": 10,
                "money_after": 1000 + index,
                "epsilon": max(1.0 - index / 100.0, 0.05) if kind == "dqn" else 0.0,
                "rebuys": 0,
            }
        )
    store.append_hands(hand_records)
    store.append_losses(
        [
            {"kind": "dqn", "step": step, "loss": 1.0 / (step + 1), "buffer": 100}
            for step in range(50)
        ]
    )
    store.save_config({"agents": "dqn=1,random=1"})
    return store


class TestRollingMean(unittest.TestCase):
    def test_rolling_mean_partial_window(self):
        self.assertEqual(rolling_mean([1.0, 3.0], 5), [1.0, 2.0])

    def test_rolling_mean_full_window(self):
        means = rolling_mean([1.0, 1.0, 4.0, 4.0], 2)
        self.assertEqual(means, [1.0, 1.0, 2.5, 4.0])


class TestRenderAll(unittest.TestCase):
    def test_render_all_writes_pngs(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            store = build_synthetic_run(temp_dir)

            rendered = render_all(store)

            self.assertEqual(len(rendered), 6)
            for path in rendered:
                self.assertTrue(os.path.exists(path))
                self.assertGreater(os.path.getsize(path), 0)

    def test_render_all_empty_run(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            store = RunStore(base_dir=temp_dir, run_name="empty-run")
            self.assertEqual(render_all(store), [])


if __name__ == "__main__":
    unittest.main()
