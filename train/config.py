import argparse
from typing import Dict, List, Tuple

from project import PlayerType

LEARNING_KINDS = ["dqn", "mc"]
HEURISTIC_KINDS: Dict[str, PlayerType] = {
    "noob": PlayerType.NOOB,
    "apprehensive": PlayerType.APPREHENSIVE,
    "aggressive": PlayerType.AGGRESSIVE,
    "random": PlayerType.RANDOM,
}


def parse_agents(agents_spec: str) -> List[Tuple[str, int]]:
    seat_counts = []
    for part in agents_spec.split(","):
        kind, _, count_text = part.strip().partition("=")
        kind = kind.strip().lower()
        if kind not in LEARNING_KINDS and kind not in HEURISTIC_KINDS:
            raise Exception(f"Unknown agent kind: {kind}")
        count = int(count_text) if count_text else 1
        if count < 1:
            raise Exception(f"Agent count must be positive: {part}")
        seat_counts.append((kind, count))
    return seat_counts


class TrainingConfig:
    def __init__(self, **kwargs):
        self.agents = kwargs.get("agents", "dqn=2,mc=1,random=1")
        self.workers = kwargs.get("workers", 0)
        self.tables = kwargs.get("tables", 1)
        self.hands = kwargs.get("hands", 10000)
        self.seed = kwargs.get("seed", 7)
        self.learning_rate = kwargs.get("learning_rate", 0.001)
        self.gamma = kwargs.get("gamma", 1.0)
        self.epsilon_start = kwargs.get("epsilon_start", 1.0)
        self.epsilon_end = kwargs.get("epsilon_end", 0.05)
        self.epsilon_decay_hands = kwargs.get("epsilon_decay_hands", 0)
        self.buffer_size = kwargs.get("buffer_size", 20000)
        self.batch_size = kwargs.get("batch_size", 64)
        self.target_sync_interval = kwargs.get("target_sync_interval", 250)
        self.sync_interval = kwargs.get("sync_interval", 25)
        self.train_interval = kwargs.get("train_interval", 4)
        self.hidden_sizes = kwargs.get("hidden_sizes", [32, 32])
        self.run_dir = kwargs.get("run_dir", "runs")
        self.starting_money = kwargs.get("starting_money", 1000)
        self.minimum_bet = kwargs.get("minimum_bet", 10)
        self.num_decks = kwargs.get("num_decks", 4)
        self.seat_counts = parse_agents(self.agents)
        self.validate()

    def total_seats(self) -> int:
        return sum(count for _, count in self.seat_counts)

    def expanded_seats(self) -> List[str]:
        seats = []
        for kind, count in self.seat_counts:
            seats.extend([kind] * count)
        return seats

    def learning_kinds_in_use(self) -> List[str]:
        return [
            kind
            for kind in LEARNING_KINDS
            if any(seat_kind == kind for seat_kind, _ in self.seat_counts)
        ]

    def seats_for_kind(self, kind: str) -> int:
        return sum(count for seat_kind, count in self.seat_counts if seat_kind == kind)

    def decay_hands_for_kind(self, kind: str) -> int:
        if self.epsilon_decay_hands > 0:
            return self.epsilon_decay_hands
        kind_hands = self.hands * self.seats_for_kind(kind) / self.total_seats()
        return max(int(kind_hands * 0.6), 1)

    def to_dict(self) -> dict:
        return {
            "agents": self.agents,
            "workers": self.workers,
            "tables": self.tables,
            "hands": self.hands,
            "seed": self.seed,
            "learning_rate": self.learning_rate,
            "gamma": self.gamma,
            "epsilon_start": self.epsilon_start,
            "epsilon_end": self.epsilon_end,
            "epsilon_decay_hands": self.epsilon_decay_hands,
            "buffer_size": self.buffer_size,
            "batch_size": self.batch_size,
            "target_sync_interval": self.target_sync_interval,
            "sync_interval": self.sync_interval,
            "train_interval": self.train_interval,
            "hidden_sizes": self.hidden_sizes,
            "run_dir": self.run_dir,
            "starting_money": self.starting_money,
            "minimum_bet": self.minimum_bet,
            "num_decks": self.num_decks,
        }

    def validate(self) -> None:
        total = self.total_seats()
        if self.tables < 1:
            raise Exception("At least one table is required")
        if self.workers < 0:
            raise Exception("Workers must be zero or more")
        if self.hands < 1:
            raise Exception("At least one hand is required")
        if total < self.tables:
            raise Exception("Every table needs at least one player")
        if total > self.tables * 4:
            raise Exception("Tables seat at most four players")
        if self.workers > self.tables:
            raise Exception("Workers cannot outnumber tables")


def parse_hidden_sizes(text: str) -> List[int]:
    return [int(part) for part in text.split(",") if part.strip()]


def config_from_args(argv=None) -> TrainingConfig:
    parser = argparse.ArgumentParser(
        description="Train blackjack agents with the from-scratch neural network"
    )
    parser.add_argument("--agents", default="dqn=2,mc=1,random=1")
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--tables", type=int, default=1)
    parser.add_argument("--hands", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--lr", dest="learning_rate", type=float, default=0.001)
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--epsilon-start", type=float, default=1.0)
    parser.add_argument("--epsilon-end", type=float, default=0.05)
    parser.add_argument("--epsilon-decay-hands", type=int, default=0)
    parser.add_argument("--buffer-size", type=int, default=20000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument(
        "--target-sync", dest="target_sync_interval", type=int, default=250
    )
    parser.add_argument("--sync-interval", type=int, default=25)
    parser.add_argument("--train-interval", type=int, default=4)
    parser.add_argument("--hidden-sizes", default="32,32")
    parser.add_argument("--run-dir", default="runs")
    parser.add_argument("--starting-money", type=int, default=1000)
    parser.add_argument("--minimum-bet", type=int, default=10)
    parser.add_argument("--num-decks", type=int, default=4)
    args = parser.parse_args(argv)

    return TrainingConfig(
        agents=args.agents,
        workers=args.workers,
        tables=args.tables,
        hands=args.hands,
        seed=args.seed,
        learning_rate=args.learning_rate,
        gamma=args.gamma,
        epsilon_start=args.epsilon_start,
        epsilon_end=args.epsilon_end,
        epsilon_decay_hands=args.epsilon_decay_hands,
        buffer_size=args.buffer_size,
        batch_size=args.batch_size,
        target_sync_interval=args.target_sync_interval,
        sync_interval=args.sync_interval,
        train_interval=args.train_interval,
        hidden_sizes=parse_hidden_sizes(args.hidden_sizes),
        run_dir=args.run_dir,
        starting_money=args.starting_money,
        minimum_bet=args.minimum_bet,
        num_decks=args.num_decks,
    )
