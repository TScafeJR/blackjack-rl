from typing import Dict, List, Optional

from project import Observation, PendingTurn, PlayerDecision, TrainTable

ACTION_DECISIONS = [
    PlayerDecision.HIT,
    PlayerDecision.STAY,
    PlayerDecision.DOUBLE_DOWN,
]
ACTION_COUNT = len(ACTION_DECISIONS)
FEATURE_COUNT = 4


def encode_observation(observation: Observation) -> List[float]:
    return [
        observation.player_total / 21.0,
        1.0 if observation.is_soft else 0.0,
        observation.dealer_upcard_value / 11.0,
        1.0 if observation.can_double else 0.0,
    ]


def encode_legal_actions(legal_actions: List[PlayerDecision]) -> List[int]:
    return [ACTION_DECISIONS.index(decision) for decision in legal_actions]


class EpisodeStep:
    def __init__(self, **kwargs):
        self.features = kwargs.get("features", [])
        self.action_index = kwargs.get("action_index", 0)
        self.legal_action_indices = kwargs.get("legal_action_indices", [])


class Episode:
    def __init__(self, **kwargs):
        self.player_id = kwargs.get("player_id", "")
        self.agent_kind = kwargs.get("agent_kind", "")
        self.steps: List[EpisodeStep] = kwargs.get("steps", [])
        self.reward = kwargs.get("reward", 0.0)
        self.outcome = kwargs.get("outcome", None)


class BlackjackEnvironment:
    def __init__(self, **kwargs):
        self.table: TrainTable = kwargs.get("table")
        self.policies: Dict[str, object] = kwargs.get("policies", {})
        self.agent_kinds: Dict[str, str] = kwargs.get("agent_kinds", {})

    def handle_pending_turn(
        self, pending_turn: PendingTurn, episodes: Dict[str, Episode]
    ) -> None:
        features = encode_observation(pending_turn.observation)
        legal_action_indices = encode_legal_actions(pending_turn.legal_actions)
        policy = self.policies[pending_turn.player_id]
        action_index = policy.select_action(pending_turn, features)
        episodes[pending_turn.player_id].steps.append(
            EpisodeStep(
                features=features,
                action_index=action_index,
                legal_action_indices=legal_action_indices,
            )
        )
        self.table.apply_decision(
            pending_turn.player_id, ACTION_DECISIONS[action_index]
        )

    def play_hand(self) -> List[Episode]:
        self.table.begin_hand()
        episodes: Dict[str, Episode] = {}
        for player in self.table.turn_order:
            episodes[player.player_id] = Episode(
                player_id=player.player_id,
                agent_kind=self.agent_kinds.get(player.player_id, ""),
                steps=[],
            )

        while True:
            pending_turn: Optional[PendingTurn] = self.table.get_pending_turn()
            if pending_turn is None:
                break
            self.handle_pending_turn(pending_turn, episodes)

        outcomes = self.table.settle_hand()
        for player_id, outcome in outcomes.items():
            episode = episodes[player_id]
            episode.reward = outcome.reward / outcome.bet
            episode.outcome = outcome
        return list(episodes.values())
