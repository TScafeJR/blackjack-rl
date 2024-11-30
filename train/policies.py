import random
from typing import List

from project import DecisionInfo, Observation, PendingTurn, Player, PlayerType

from .environment import ACTION_DECISIONS, encode_observation


class BasePolicy:
    uses_count = False

    def __init__(self, **kwargs):
        self.uses_count = kwargs.get("uses_count", False)

    def encode(self, observation: Observation) -> List[float]:
        return encode_observation(observation, self.uses_count)

    def select_action(self, pending_turn: PendingTurn, features: List[float]) -> int:
        raise Exception("Select action is not implemented")

    def is_learning(self) -> bool:
        return False


class NetworkPolicy(BasePolicy):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.network = kwargs.get("network")
        self.epsilon = kwargs.get("epsilon", 0.0)
        self.rng = kwargs.get("rng", random.Random())

    def select_action(self, pending_turn: PendingTurn, features: List[float]) -> int:
        legal_action_indices = [
            ACTION_DECISIONS.index(decision) for decision in pending_turn.legal_actions
        ]
        if self.rng.random() < self.epsilon:
            return self.rng.choice(legal_action_indices)

        q_values = self.network.forward([features])[0]
        best_index = legal_action_indices[0]
        for action_index in legal_action_indices:
            if q_values[action_index] > q_values[best_index]:
                best_index = action_index
        return best_index

    def apply_snapshot(self, snapshot: dict) -> None:
        self.network.set_weights(snapshot["weights"])
        self.epsilon = snapshot["epsilon"]

    def is_learning(self) -> bool:
        return True


class HeuristicPolicy(BasePolicy):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.player_type: PlayerType = kwargs.get("player_type", PlayerType.RANDOM)
        self.rng = kwargs.get("rng", random.Random())

    def select_action(self, pending_turn: PendingTurn, features: List[float]) -> int:
        legal_action_indices = [
            ACTION_DECISIONS.index(decision) for decision in pending_turn.legal_actions
        ]
        decision_info = DecisionInfo(
            observation=pending_turn.observation,
            legal_actions=pending_turn.legal_actions,
        )
        decision = Player.decide_for_type(self.player_type, decision_info)
        if decision in ACTION_DECISIONS:
            action_index = ACTION_DECISIONS.index(decision)
            if action_index in legal_action_indices:
                return action_index
        return self.rng.choice(legal_action_indices)
