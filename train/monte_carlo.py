from typing import List

from .base_learner import BaseLearner
from .environment import Episode
from .replay_buffer import Transition


class MonteCarloLearner(BaseLearner):
    def handle_episode(self, episode: Episode) -> None:
        episode_return = episode.reward
        for step in reversed(episode.steps):
            self.buffer.push(
                Transition(
                    features=step.features,
                    action_index=step.action_index,
                    reward=episode_return,
                    done=True,
                )
            )
            episode_return *= self.gamma

    def compute_target_values(self, batch: List[Transition]) -> List[float]:
        return [transition.reward for transition in batch]
