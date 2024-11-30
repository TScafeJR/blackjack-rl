import random
import tempfile
import unittest

from train.base_learner import build_network
from train.betting import BET_WEIGHTS_SUFFIX
from train.environment import (BET_ACTION_COUNT, BET_FEATURE_COUNT, BET_UNITS,
                               feature_count_for)

from .board import ACTION_NAMES, BoardSession
from .run_store import RunStore, latest_run_path


def build_run_with_weights(temp_dir):
    store = RunStore(base_dir=temp_dir, run_name="board-run")
    network = build_network([32, 32], random.Random(1))
    store.save_weights("dqn", network.get_weights())
    return store


def build_run_with_ramp_weights(temp_dir):
    store = build_run_with_weights(temp_dir)
    store.save_weights(
        "dqn-ramp",
        build_network(
            [32, 32], random.Random(2), feature_count_for(True)
        ).get_weights(),
    )
    store.save_weights(
        f"dqn-ramp{BET_WEIGHTS_SUFFIX}",
        build_network(
            [32, 32], random.Random(3), BET_FEATURE_COUNT, BET_ACTION_COUNT
        ).get_weights(),
    )
    return store


class TestBoardSession(unittest.TestCase):
    def test_play_hand_events_structure(self):
        random.seed(17)
        with tempfile.TemporaryDirectory() as temp_dir:
            store = build_run_with_weights(temp_dir)
            session = BoardSession(
                run_path=store.run_path, agents="dqn=1,random=1", seed=17
            )

            events = session.play_hand_events()

            self.assertEqual(events[0]["type"], "deal")
            self.assertEqual(len(events[0]["players"]), 2)
            self.assertIn(" of ", events[0]["dealer_upcard"])
            self.assertEqual(events[-1]["type"], "settle")
            self.assertEqual(len(events[-1]["results"]), 2)
            self.assertEqual(events[-2]["type"], "dealer")
            for event in events:
                if event["type"] != "decision":
                    continue
                self.assertIn(event["action"], ACTION_NAMES)
                kind = session.agent_kinds[event["player_id"]]
                if kind == "dqn":
                    self.assertEqual(sorted(event["q_values"]), sorted(ACTION_NAMES))
                else:
                    self.assertIsNone(event["q_values"])

    def test_money_updates_across_hands(self):
        random.seed(17)
        with tempfile.TemporaryDirectory() as temp_dir:
            store = build_run_with_weights(temp_dir)
            session = BoardSession(run_path=store.run_path, agents="dqn=1", seed=17)

            for _ in range(5):
                events = session.play_hand_events()
                result = events[-1]["results"][0]
                self.assertIsInstance(result["money"], int)
            self.assertEqual(session.hands_played, 5)

    def test_history_records_hands(self):
        random.seed(17)
        with tempfile.TemporaryDirectory() as temp_dir:
            store = build_run_with_weights(temp_dir)
            session = BoardSession(
                run_path=store.run_path, agents="dqn=1,random=1", seed=17
            )

            for _ in range(3):
                session.play_hand_events()

            self.assertEqual(len(session.history), 3)
            record = session.history[-1]
            self.assertEqual(record["hand_number"], 3)
            self.assertEqual(len(record["seats"]), 2)
            self.assertGreaterEqual(record["dealer_total"], 17)
            for seat in record["seats"]:
                self.assertIn("actions", seat)
                self.assertIn("result", seat)
                self.assertIsInstance(seat["money"], int)

    def test_history_respects_limit(self):
        random.seed(17)
        with tempfile.TemporaryDirectory() as temp_dir:
            store = build_run_with_weights(temp_dir)
            session = BoardSession(
                run_path=store.run_path, agents="dqn=1", seed=17, history_limit=2
            )

            for _ in range(4):
                session.play_hand_events()

            self.assertEqual(len(session.history), 2)
            self.assertEqual(session.history[-1]["hand_number"], 4)

    def test_heuristic_only_session_needs_no_weights(self):
        random.seed(17)
        with tempfile.TemporaryDirectory() as temp_dir:
            session = BoardSession(run_path=temp_dir, agents="random=1", seed=17)
            events = session.play_hand_events()
            self.assertEqual(events[0]["type"], "deal")

    def test_missing_weights_raises(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            with self.assertRaises(Exception):
                BoardSession(run_path=temp_dir, agents="dqn=1", seed=17)


class TestBoardCounting(unittest.TestCase):
    def test_ramp_seat_sizes_its_own_bet(self):
        random.seed(17)
        with tempfile.TemporaryDirectory() as temp_dir:
            store = build_run_with_ramp_weights(temp_dir)
            session = BoardSession(
                run_path=store.run_path, agents="dqn-ramp=1,dqn=1", seed=17
            )

            deal = session.play_hand_events()[0]

            self.assertIn("true_count", deal)
            bets = sorted(deal["bets"].values())
            self.assertEqual(bets[0], 10)
            self.assertIn(bets[-1] // 10, BET_UNITS)

    def test_ramp_seat_needs_bet_weights(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            store = build_run_with_weights(temp_dir)
            store.save_weights(
                "dqn-ramp",
                build_network(
                    [32, 32], random.Random(2), feature_count_for(True)
                ).get_weights(),
            )
            with self.assertRaises(Exception):
                BoardSession(run_path=store.run_path, agents="dqn-ramp=1", seed=17)

    def test_dealer_rule_is_configurable(self):
        random.seed(17)
        with tempfile.TemporaryDirectory() as temp_dir:
            session = BoardSession(
                run_path=temp_dir, agents="basic=1", seed=17, dealer_rule="s17"
            )

            for _ in range(20):
                events = session.play_hand_events()
                dealer = [e for e in events if e["type"] == "dealer"][0]
                if not dealer["is_bust"]:
                    self.assertGreaterEqual(dealer["total"], 17)
            self.assertEqual(session.describe()["dealer_rule"], "s17")


class TestLatestRunPath(unittest.TestCase):
    def test_latest_run_path_picks_newest(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            RunStore(base_dir=temp_dir, run_name="20240901-100000")
            RunStore(base_dir=temp_dir, run_name="20240928-190500")
            self.assertTrue(latest_run_path(temp_dir).endswith("20240928-190500"))

    def test_latest_run_path_empty_raises(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            with self.assertRaises(Exception):
                latest_run_path(temp_dir)


if __name__ == "__main__":
    unittest.main()
