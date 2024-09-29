import os
import random
import tempfile
import unittest

from train.base_learner import build_network

from .plots import (build_basic_grid, build_policy_grid, render_all,
                    render_network_plots, rolling_mean)
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

    def test_render_all_includes_network_plots_with_weights(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            store = build_synthetic_run(temp_dir)
            store.save_weights(
                "dqn", build_network([32, 32], random.Random(1)).get_weights()
            )
            store.save_weights(
                "mc", build_network([32, 32], random.Random(2)).get_weights()
            )

            rendered = render_all(store)

            self.assertEqual(len(rendered), 9)
            names = [os.path.basename(path) for path in rendered]
            self.assertIn("network_weights.png", names)
            self.assertIn("policy_charts.png", names)
            self.assertIn("policy_vs_basic.png", names)


class TestNetworkPlots(unittest.TestCase):
    def test_render_network_plots_without_weights(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            store = RunStore(base_dir=temp_dir, run_name="no-weights")
            self.assertEqual(render_network_plots(store), [])

    def test_build_basic_grid_matches_known_cells(self):
        hard_grid = build_basic_grid(soft=False)
        soft_grid = build_basic_grid(soft=True)

        self.assertEqual(hard_grid[11 - 4][11 - 2], 2)
        self.assertEqual(hard_grid[16 - 4][10 - 2], 0)
        self.assertEqual(hard_grid[16 - 4][6 - 2], 1)
        self.assertEqual(hard_grid[17 - 4][11 - 2], 1)
        self.assertEqual(soft_grid[18 - 12][3 - 2], 2)
        self.assertEqual(soft_grid[20 - 12][6 - 2], 1)

    def test_build_policy_grid_shapes(self):
        network = build_network([32, 32], random.Random(1))
        network.set_training(False)

        hard_grid = build_policy_grid(network, soft=False)
        soft_grid = build_policy_grid(network, soft=True)

        self.assertEqual(len(hard_grid), 18)
        self.assertEqual(len(soft_grid), 10)
        for row in hard_grid + soft_grid:
            self.assertEqual(len(row), 10)
            for action_index in row:
                self.assertIn(action_index, [0, 1, 2])


if __name__ == "__main__":
    unittest.main()
