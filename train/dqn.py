from typing import List

from .base_learner import BaseLearner, build_network
from .environment import Episode
from .replay_buffer import Transition


class DQNLearner(BaseLearner):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.target_sync_interval = kwargs.get("target_sync_interval", 500)
        self.target_network = build_network(
            self.hidden_sizes, self.rng, self.feature_count, self.action_count
        )
        self.sync_target_network()

    def sync_target_network(self) -> None:
        self.target_network.set_weights(self.network.get_weights())

    def handle_episode(self, episode: Episode) -> None:
        steps = episode.steps
        for step_index, step in enumerate(steps):
            is_last = step_index == len(steps) - 1
            next_step = None if is_last else steps[step_index + 1]
            self.buffer.push(
                Transition(
                    features=step.features,
                    action_index=step.action_index,
                    reward=episode.reward if is_last else 0.0,
                    next_features=None if is_last else next_step.features,
                    next_legal_action_indices=(
                        [] if is_last else next_step.legal_action_indices
                    ),
                    done=is_last,
                )
            )

    def compute_target_values(self, batch: List[Transition]) -> List[float]:
        target_values = []
        for transition in batch:
            if transition.done:
                target_values.append(transition.reward)
                continue
            next_q_values = self.target_network.forward([transition.next_features])[0]
            best_next = max(
                next_q_values[action_index]
                for action_index in transition.next_legal_action_indices
            )
            target_values.append(transition.reward + self.gamma * best_next)
        return target_values

    def handle_post_step(self) -> None:
        if self.train_steps % self.target_sync_interval == 0:
            self.sync_target_network()
