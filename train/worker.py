import queue
import random
from collections import deque
from typing import Dict, List

from project import Dealer, Player, PlayerType, TrainTable

from .base_learner import build_network
from .config import HEURISTIC_KINDS, LEARNING_KINDS
from .environment import BlackjackEnvironment, Episode
from .policies import BasePolicy, HeuristicPolicy, NetworkPolicy


class FakeQueue:
    def __init__(self):
        self.items = deque()

    def put(self, item) -> None:
        self.items.append(item)

    def get_nowait(self):
        if not self.items:
            raise queue.Empty
        return self.items.popleft()


class FakeEvent:
    def __init__(self):
        self.flagged = False

    def set(self) -> None:
        self.flagged = True

    def is_set(self) -> bool:
        return self.flagged


class WorkerSpec:
    def __init__(self, **kwargs):
        self.worker_id = kwargs.get("worker_id", 0)
        self.seed = kwargs.get("seed", 7)
        self.hand_budget = kwargs.get("hand_budget", 1000)
        self.flush_rounds = kwargs.get("flush_rounds", 25)
        self.table_seats: List[List[str]] = kwargs.get("table_seats", [])
        self.hidden_sizes = kwargs.get("hidden_sizes", [32, 32])
        self.starting_money = kwargs.get("starting_money", 1000)
        self.minimum_bet = kwargs.get("minimum_bet", 10)
        self.num_decks = kwargs.get("num_decks", 4)


class TableWorker:
    def __init__(self, spec: WorkerSpec, experience_queue, weights_queue, stop_event):
        self.spec = spec
        self.experience_queue = experience_queue
        self.weights_queue = weights_queue
        self.stop_event = stop_event
        self.environments: List[BlackjackEnvironment] = []
        self.network_policies: Dict[str, NetworkPolicy] = {}
        self.table_by_player: Dict[str, int] = {}
        self.pending_episodes: List[Episode] = []
        self.pending_records: List[dict] = []
        self.hands_done = 0
        self.rounds_since_flush = 0
        self.setup()

    def build_policy(self, kind: str) -> BasePolicy:
        if kind in LEARNING_KINDS:
            if kind not in self.network_policies:
                network = build_network(
                    self.spec.hidden_sizes, random.Random(self.spec.seed)
                )
                network.set_training(False)
                self.network_policies[kind] = NetworkPolicy(
                    network=network, epsilon=1.0, rng=random.Random(self.spec.seed)
                )
            return self.network_policies[kind]
        return HeuristicPolicy(
            player_type=HEURISTIC_KINDS[kind], rng=random.Random(self.spec.seed)
        )

    def build_environment(self, table_index: int, seat_kinds: List[str]) -> None:
        table = TrainTable(
            num_decks=self.spec.num_decks,
            minimum_bet=self.spec.minimum_bet,
            rebuy=True,
        )
        table.add_dealer(Dealer())
        policies: Dict[str, BasePolicy] = {}
        agent_kinds: Dict[str, str] = {}
        for kind in seat_kinds:
            player = Player(
                starting_money=self.spec.starting_money,
                player_type=PlayerType.RANDOM,
            )
            table.add_player(player)
            policies[player.player_id] = self.build_policy(kind)
            agent_kinds[player.player_id] = kind
            self.table_by_player[player.player_id] = table_index
        self.environments.append(
            BlackjackEnvironment(
                table=table, policies=policies, agent_kinds=agent_kinds
            )
        )

    def setup(self) -> None:
        random.seed(self.spec.seed)
        for table_index, seat_kinds in enumerate(self.spec.table_seats):
            self.build_environment(table_index, seat_kinds)

    def apply_weights(self) -> None:
        latest_snapshot_map = None
        while True:
            try:
                latest_snapshot_map = self.weights_queue.get_nowait()
            except queue.Empty:
                break
        if latest_snapshot_map is None:
            return
        for kind, snapshot in latest_snapshot_map.items():
            if kind in self.network_policies:
                self.network_policies[kind].apply_snapshot(snapshot)

    def build_record(self, environment: BlackjackEnvironment, episode: Episode) -> dict:
        policy = environment.policies[episode.player_id]
        epsilon = policy.epsilon if isinstance(policy, NetworkPolicy) else 0.0
        return {
            "worker_id": self.spec.worker_id,
            "table_index": self.table_by_player[episode.player_id],
            "player_id": episode.player_id,
            "kind": episode.agent_kind,
            "result": episode.outcome.result.name,
            "reward": episode.reward,
            "bet": episode.outcome.bet,
            "money_after": episode.outcome.money_after,
            "epsilon": epsilon,
            "rebuys": environment.table.get_rebuys(episode.player_id),
            "hand_index": self.hands_done,
        }

    def play_iteration(self) -> int:
        hands_this_round = 0
        for environment in self.environments:
            episodes = environment.play_hand()
            for episode in episodes:
                self.hands_done += 1
                hands_this_round += 1
                self.pending_records.append(self.build_record(environment, episode))
                if episode.agent_kind in LEARNING_KINDS and episode.steps:
                    self.pending_episodes.append(episode)
        self.rounds_since_flush += 1
        return hands_this_round

    def flush(self) -> None:
        if not self.pending_records:
            self.rounds_since_flush = 0
            return
        self.experience_queue.put(
            {
                "worker_id": self.spec.worker_id,
                "episodes": self.pending_episodes,
                "records": self.pending_records,
            }
        )
        self.pending_episodes = []
        self.pending_records = []
        self.rounds_since_flush = 0

    def run(self) -> None:
        while not self.stop_event.is_set() and self.hands_done < self.spec.hand_budget:
            self.apply_weights()
            self.play_iteration()
            if self.rounds_since_flush >= self.spec.flush_rounds:
                self.flush()
        self.flush()


def run_worker(spec: WorkerSpec, experience_queue, weights_queue, stop_event) -> None:
    worker = TableWorker(spec, experience_queue, weights_queue, stop_event)
    worker.run()
