import queue
from typing import Dict, List

from review import MetricsCollector, RunStore

from .base_learner import BaseLearner


class LearnerLoop:
    def __init__(self, **kwargs):
        self.learners: Dict[str, BaseLearner] = kwargs.get("learners", {})
        self.metrics: MetricsCollector = kwargs.get("metrics")
        self.run_store: RunStore = kwargs.get("run_store")
        self.train_interval = kwargs.get("train_interval", 4)
        self.max_debt = kwargs.get("max_debt", 200)
        self.train_debt = 0.0

    def process_message(self, message: dict) -> int:
        episodes_by_kind: Dict[str, List] = {}
        for episode in message["episodes"]:
            episodes_by_kind.setdefault(episode.agent_kind, []).append(episode)
        for kind, episodes in episodes_by_kind.items():
            self.learners[kind].ingest(episodes)
        records = message["records"]
        self.metrics.record_hands(records)
        self.run_store.append_hands(records)
        return len(records)

    def drain(self, experience_queue) -> int:
        hands_ingested = 0
        while True:
            try:
                message = experience_queue.get_nowait()
            except queue.Empty:
                break
            hands_ingested += self.process_message(message)
        return hands_ingested

    def train(self, hands_ingested: int) -> None:
        self.train_debt = min(
            self.train_debt + hands_ingested / self.train_interval, self.max_debt
        )
        loss_records = []
        while self.train_debt >= 1.0:
            self.train_debt -= 1.0
            for kind, learner in self.learners.items():
                loss_value = learner.train_step()
                if loss_value is None:
                    continue
                loss_record = {
                    "kind": kind,
                    "step": learner.train_steps,
                    "loss": loss_value,
                    "buffer": len(learner.buffer),
                    "hands_seen": learner.hands_seen,
                }
                self.metrics.record_loss(loss_record)
                loss_records.append(loss_record)
        self.run_store.append_losses(loss_records)

    def broadcast(self, weights_queues: List) -> None:
        snapshot_map = {
            kind: learner.get_snapshot() for kind, learner in self.learners.items()
        }
        for weights_queue in weights_queues:
            weights_queue.put(snapshot_map)
