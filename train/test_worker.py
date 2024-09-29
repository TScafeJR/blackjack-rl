import queue
import unittest

from .dqn import DQNLearner
from .worker import FakeEvent, FakeQueue, TableWorker, WorkerSpec


def build_worker():
    spec = WorkerSpec(
        worker_id=0,
        seed=11,
        hand_budget=50,
        flush_rounds=5,
        table_seats=[["dqn", "noob"]],
        hidden_sizes=[32, 32],
        starting_money=1000,
        minimum_bet=10,
        num_decks=4,
    )
    experience_queue = FakeQueue()
    weights_queue = FakeQueue()
    stop_event = FakeEvent()
    worker = TableWorker(spec, experience_queue, weights_queue, stop_event)
    return worker, experience_queue, weights_queue, stop_event


class TestFakeChannels(unittest.TestCase):
    def test_fake_queue(self):
        fake_queue = FakeQueue()
        fake_queue.put("first")
        fake_queue.put("second")
        self.assertEqual(fake_queue.get_nowait(), "first")
        self.assertEqual(fake_queue.get_nowait(), "second")
        with self.assertRaises(queue.Empty):
            fake_queue.get_nowait()

    def test_fake_event(self):
        fake_event = FakeEvent()
        self.assertFalse(fake_event.is_set())
        fake_event.set()
        self.assertTrue(fake_event.is_set())


class TestTableWorker(unittest.TestCase):
    def test_play_iteration_collects_records(self):
        worker, _, _, _ = build_worker()

        hands = worker.play_iteration()

        self.assertEqual(hands, 2)
        self.assertEqual(len(worker.pending_records), 2)
        kinds = sorted(record["kind"] for record in worker.pending_records)
        self.assertEqual(kinds, ["dqn", "noob"])
        self.assertEqual(len(worker.pending_episodes), 1)

    def test_flush_sends_message(self):
        worker, experience_queue, _, _ = build_worker()
        worker.play_iteration()

        worker.flush()

        message = experience_queue.get_nowait()
        self.assertEqual(message["worker_id"], 0)
        self.assertEqual(len(message["records"]), 2)
        self.assertEqual(worker.pending_records, [])
        with self.assertRaises(queue.Empty):
            experience_queue.get_nowait()

    def test_apply_weights_uses_latest_snapshot(self):
        worker, _, weights_queue, _ = build_worker()
        learner = DQNLearner(seed=2, hidden_sizes=[32, 32])
        stale_snapshot = learner.get_snapshot()
        stale_snapshot["epsilon"] = 0.9
        fresh_snapshot = learner.get_snapshot()
        fresh_snapshot["epsilon"] = 0.25
        weights_queue.put({"dqn": stale_snapshot})
        weights_queue.put({"dqn": fresh_snapshot})

        worker.apply_weights()

        policy = worker.network_policies["dqn"]
        self.assertEqual(policy.epsilon, 0.25)
        self.assertEqual(policy.network.get_weights(), learner.network.get_weights())

    def test_run_respects_budget(self):
        worker, experience_queue, _, _ = build_worker()

        worker.run()

        self.assertGreaterEqual(worker.hands_done, 50)
        total_records = 0
        while True:
            try:
                message = experience_queue.get_nowait()
            except queue.Empty:
                break
            total_records += len(message["records"])
        self.assertEqual(total_records, worker.hands_done)


if __name__ == "__main__":
    unittest.main()
