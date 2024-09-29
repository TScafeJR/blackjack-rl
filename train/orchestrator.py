import multiprocessing
import queue
import random
import time
from typing import Dict, List

from review import MetricsCollector, RunStore, format_summary, summarize_hands
from review.plots import render_all

from .base_learner import BaseLearner
from .config import TrainingConfig
from .dqn import DQNLearner
from .learner import LearnerLoop
from .monte_carlo import MonteCarloLearner
from .worker import FakeEvent, FakeQueue, TableWorker, WorkerSpec, run_worker


class TrainingRun:
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.metrics = MetricsCollector()
        self.run_store: RunStore = None
        self.learners: Dict[str, BaseLearner] = {}
        self.started_at = 0.0

    def build_learner(self, kind: str) -> BaseLearner:
        learner_kwargs = {
            "hidden_sizes": self.config.hidden_sizes,
            "learning_rate": self.config.learning_rate,
            "gamma": self.config.gamma,
            "epsilon_start": self.config.epsilon_start,
            "epsilon_end": self.config.epsilon_end,
            "epsilon_decay_hands": self.config.decay_hands_for_kind(kind),
            "buffer_size": self.config.buffer_size,
            "batch_size": self.config.batch_size,
            "target_sync_interval": self.config.target_sync_interval,
            "seed": self.config.seed,
        }
        if kind == "dqn":
            return DQNLearner(**learner_kwargs)
        return MonteCarloLearner(**learner_kwargs)

    def build_learners(self) -> Dict[str, BaseLearner]:
        return {
            kind: self.build_learner(kind)
            for kind in self.config.learning_kinds_in_use()
        }

    def build_table_seats(self) -> List[List[str]]:
        table_seats: List[List[str]] = [[] for _ in range(self.config.tables)]
        for seat_index, kind in enumerate(self.config.expanded_seats()):
            table_seats[seat_index % self.config.tables].append(kind)
        return table_seats

    def build_learner_loop(self) -> LearnerLoop:
        return LearnerLoop(
            learners=self.learners,
            metrics=self.metrics,
            run_store=self.run_store,
            train_interval=self.config.train_interval,
        )

    def run_sync(self) -> None:
        experience_queue = FakeQueue()
        weights_queue = FakeQueue()
        stop_event = FakeEvent()
        spec = WorkerSpec(
            worker_id=0,
            seed=self.config.seed,
            hand_budget=self.config.hands,
            flush_rounds=self.config.sync_interval,
            table_seats=self.build_table_seats(),
            hidden_sizes=self.config.hidden_sizes,
            starting_money=self.config.starting_money,
            minimum_bet=self.config.minimum_bet,
            num_decks=self.config.num_decks,
        )
        learner_loop = self.build_learner_loop()
        learner_loop.broadcast([weights_queue])
        worker = TableWorker(spec, experience_queue, weights_queue, stop_event)

        while worker.hands_done < self.config.hands:
            worker.apply_weights()
            worker.play_iteration()
            if worker.rounds_since_flush >= spec.flush_rounds:
                worker.flush()
                learner_loop.train(learner_loop.drain(experience_queue))
                learner_loop.broadcast([weights_queue])
        worker.flush()
        learner_loop.train(learner_loop.drain(experience_queue))

    def build_worker_specs(self) -> List[WorkerSpec]:
        table_seats = self.build_table_seats()
        total_seats = self.config.total_seats()
        specs = []
        for worker_id in range(self.config.workers):
            worker_tables = table_seats[worker_id :: self.config.workers]
            if not worker_tables:
                continue
            worker_seats = sum(len(seats) for seats in worker_tables)
            specs.append(
                WorkerSpec(
                    worker_id=worker_id,
                    seed=self.config.seed + worker_id + 1,
                    hand_budget=round(self.config.hands * worker_seats / total_seats),
                    flush_rounds=self.config.sync_interval,
                    table_seats=worker_tables,
                    hidden_sizes=self.config.hidden_sizes,
                    starting_money=self.config.starting_money,
                    minimum_bet=self.config.minimum_bet,
                    num_decks=self.config.num_decks,
                )
            )
        return specs

    @staticmethod
    def shutdown_workers(processes, stop_event, experience_queue, learner_loop):
        stop_event.set()
        for process in processes:
            attempts = 0
            while process.is_alive() and attempts < 20:
                process.join(timeout=0.25)
                learner_loop.drain(experience_queue)
                attempts += 1
            if process.is_alive():
                process.terminate()
                process.join(timeout=5)
        learner_loop.drain(experience_queue)

    def run_parallel(self) -> None:
        context = multiprocessing.get_context("spawn")
        experience_queue = context.Queue(maxsize=16)
        stop_event = context.Event()
        specs = self.build_worker_specs()
        weights_queues = [context.Queue() for _ in specs]
        learner_loop = self.build_learner_loop()
        learner_loop.broadcast(weights_queues)

        processes = [
            context.Process(
                target=run_worker,
                args=(spec, experience_queue, weights_queue, stop_event),
                daemon=True,
            )
            for spec, weights_queue in zip(specs, weights_queues)
        ]
        for process in processes:
            process.start()

        total_hands = 0
        try:
            while total_hands < self.config.hands and any(
                process.is_alive() for process in processes
            ):
                try:
                    message = experience_queue.get(timeout=0.5)
                except queue.Empty:
                    continue
                hands_ingested = learner_loop.process_message(message)
                hands_ingested += learner_loop.drain(experience_queue)
                total_hands += hands_ingested
                learner_loop.train(hands_ingested)
                learner_loop.broadcast(weights_queues)
        except KeyboardInterrupt:
            print("stopping workers after interrupt")
        finally:
            self.shutdown_workers(processes, stop_event, experience_queue, learner_loop)

    def build_report(self) -> str:
        duration = max(time.monotonic() - self.started_at, 1e-9)
        hands_recorded = len(self.metrics.hand_records)
        lines = [
            f"run: {self.run_store.run_path}",
            f"agents: {self.config.agents}",
            (
                f"tables: {self.config.tables}  workers: {self.config.workers}  "
                f"hands: {self.config.hands}  seed: {self.config.seed}"
            ),
            (
                f"played {hands_recorded} hands in {duration:.0f}s "
                f"({hands_recorded / duration:.0f} hands/s)"
            ),
        ]
        for kind, learner in self.learners.items():
            lines.append(
                f"{kind}: {learner.count_parameters()} parameters, "
                f"{learner.train_steps} train steps, "
                f"final epsilon {learner.get_epsilon():.3f}"
            )
        lines.append("")
        lines.append(format_summary(summarize_hands(self.metrics.hand_records)))
        return "\n".join(lines)

    def finalize(self) -> str:
        for kind, learner in self.learners.items():
            self.run_store.save_weights(kind, learner.network.get_weights())
        report = self.build_report()
        print(report)
        self.run_store.save_report(report)
        for plot_path in render_all(self.run_store):
            print(f"plot: {plot_path}")
        return self.run_store.run_path

    def execute(self) -> str:
        self.started_at = time.monotonic()
        random.seed(self.config.seed)
        self.run_store = RunStore(base_dir=self.config.run_dir)
        self.run_store.save_config(self.config.to_dict())
        self.learners = self.build_learners()
        if self.config.workers == 0:
            self.run_sync()
        else:
            self.run_parallel()
        return self.finalize()
