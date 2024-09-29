import random
from typing import List, Optional

from neural import Adam, Linear, MSELoss, ReLU, Sequential

from .environment import ACTION_COUNT, FEATURE_COUNT, Episode
from .replay_buffer import ReplayBuffer, Transition


def build_network(hidden_sizes: List[int], rng: random.Random) -> Sequential:
    layers = []
    in_features = FEATURE_COUNT
    for hidden_size in hidden_sizes:
        layers.append(
            Linear(in_features=in_features, out_features=hidden_size, rng=rng)
        )
        layers.append(ReLU())
        in_features = hidden_size
    layers.append(Linear(in_features=in_features, out_features=ACTION_COUNT, rng=rng))
    return Sequential(layers)


class BaseLearner:
    def __init__(self, **kwargs):
        self.rng = random.Random(kwargs.get("seed"))
        self.hidden_sizes = kwargs.get("hidden_sizes", [32, 32])
        self.gamma = kwargs.get("gamma", 1.0)
        self.epsilon_start = kwargs.get("epsilon_start", 1.0)
        self.epsilon_end = kwargs.get("epsilon_end", 0.05)
        self.epsilon_decay_hands = kwargs.get("epsilon_decay_hands", 10000)
        self.batch_size = kwargs.get("batch_size", 64)
        self.network = build_network(self.hidden_sizes, self.rng)
        self.buffer = ReplayBuffer(
            capacity=kwargs.get("buffer_size", 10000), rng=self.rng
        )
        self.loss = MSELoss()
        self.optimizer = Adam(
            self.network.get_parameters(),
            learning_rate=kwargs.get("learning_rate", 0.001),
        )
        self.hands_seen = 0
        self.train_steps = 0

    def get_epsilon(self) -> float:
        if self.hands_seen >= self.epsilon_decay_hands:
            return self.epsilon_end
        progress = self.hands_seen / self.epsilon_decay_hands
        return self.epsilon_start + (self.epsilon_end - self.epsilon_start) * progress

    def handle_episode(self, episode: Episode) -> None:
        raise Exception("Handle episode is not implemented")

    def ingest(self, episodes: List[Episode]) -> None:
        for episode in episodes:
            self.hands_seen += 1
            if episode.steps:
                self.handle_episode(episode)

    def compute_target_values(self, batch: List[Transition]) -> List[float]:
        raise Exception("Compute target values is not implemented")

    def train_step(self) -> Optional[float]:
        if len(self.buffer) < self.batch_size:
            return None
        batch = self.buffer.sample(self.batch_size)
        target_values = self.compute_target_values(batch)

        features = [transition.features for transition in batch]
        predictions = self.network.forward(features)
        targets = [list(row) for row in predictions]
        for row_index, transition in enumerate(batch):
            targets[row_index][transition.action_index] = target_values[row_index]

        loss_value = self.loss.forward(predictions, targets)
        self.optimizer.zero_grad()
        self.network.backward(self.loss.backward())
        self.optimizer.step()
        self.train_steps += 1
        self.handle_post_step()
        return loss_value

    def handle_post_step(self) -> None:
        return

    def get_snapshot(self) -> dict:
        return {"weights": self.network.get_weights(), "epsilon": self.get_epsilon()}

    def count_parameters(self) -> int:
        return self.network.count_parameters()
