import random
from typing import List

from .base_learner import BaseLearner, build_network
from .environment import (BET_ACTION_COUNT, BET_FEATURE_COUNT, BET_UNITS,
                          Episode)
from .replay_buffer import Transition

BET_WEIGHTS_SUFFIX = "-bet"


class BetPolicy:
    def __init__(self, **kwargs):
        self.network = kwargs.get("network")
        self.epsilon = kwargs.get("epsilon", 0.0)
        self.rng = kwargs.get("rng", random.Random())

    def select_action(self, features: List[float]) -> int:
        if self.rng.random() < self.epsilon:
            return self.rng.randrange(BET_ACTION_COUNT)

        values = self.network.forward([features])[0]
        return values.index(max(values))

    def units_for(self, features: List[float]) -> int:
        return BET_UNITS[self.select_action(features)]

    def apply_snapshot(self, snapshot: dict) -> None:
        self.network.set_weights(snapshot["weights"])
        self.epsilon = snapshot["epsilon"]


class BetLearner(BaseLearner):
    def __init__(self, **kwargs):
        kwargs.setdefault("feature_count", BET_FEATURE_COUNT)
        kwargs.setdefault("action_count", BET_ACTION_COUNT)
        super().__init__(**kwargs)

    @staticmethod
    def should_handle(episode: Episode) -> bool:
        return episode.bet_features is not None

    def handle_episode(self, episode: Episode) -> None:
        self.buffer.push(
            Transition(
                features=episode.bet_features,
                action_index=episode.bet_action_index,
                reward=episode.bet_reward,
                done=True,
            )
        )

    def compute_target_values(self, batch: List[Transition]) -> List[float]:
        return [transition.reward for transition in batch]


def build_bet_policy(hidden_sizes: List[int], rng: random.Random) -> BetPolicy:
    network = build_network(hidden_sizes, rng, BET_FEATURE_COUNT, BET_ACTION_COUNT)
    network.set_training(False)
    return BetPolicy(network=network, epsilon=1.0, rng=rng)
