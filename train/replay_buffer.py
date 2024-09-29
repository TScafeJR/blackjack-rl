import random
from typing import List


class Transition:
    def __init__(self, **kwargs):
        self.features = kwargs.get("features", [])
        self.action_index = kwargs.get("action_index", 0)
        self.reward = kwargs.get("reward", 0.0)
        self.next_features = kwargs.get("next_features", None)
        self.next_legal_action_indices = kwargs.get("next_legal_action_indices", [])
        self.done = kwargs.get("done", True)


class ReplayBuffer:
    def __init__(self, **kwargs):
        self.capacity = kwargs.get("capacity", 10000)
        self.rng = kwargs.get("rng", random.Random())
        self.transitions: List[Transition] = []
        self.write_index = 0

    def push(self, transition: Transition) -> None:
        if len(self.transitions) < self.capacity:
            self.transitions.append(transition)
            return
        self.transitions[self.write_index] = transition
        self.write_index = (self.write_index + 1) % self.capacity

    def sample(self, batch_size: int) -> List[Transition]:
        return self.rng.sample(self.transitions, batch_size)

    def __len__(self) -> int:
        return len(self.transitions)
