from typing import Dict, List, Optional, Tuple

from project import (BET_UNIT_CAP, Observation, PendingTurn, PlayerDecision,
                     TrainTable)

ACTION_DECISIONS = [
    PlayerDecision.HIT,
    PlayerDecision.STAY,
    PlayerDecision.DOUBLE_DOWN,
]
ACTION_COUNT = len(ACTION_DECISIONS)
FEATURE_COUNT = 4
COUNT_FEATURE_COUNT = FEATURE_COUNT + 1
COUNT_SCALE = 10.0

BET_UNITS = list(range(1, BET_UNIT_CAP + 1))
BET_ACTION_COUNT = len(BET_UNITS)
BET_FEATURE_COUNT = 2
DECK_SCALE = 8.0

FINAL_BET_SCALE = "final_bet"
INITIAL_BET_SCALE = "initial_bet"
REWARD_SCALES = [FINAL_BET_SCALE, INITIAL_BET_SCALE]


def scale_count(true_count: float) -> float:
    return max(-1.0, min(1.0, true_count / COUNT_SCALE))


def bet_action_for_units(units: int) -> int:
    if units in BET_UNITS:
        return BET_UNITS.index(units)
    return 0


def encode_bet_state(true_count: float, decks_remaining: float) -> List[float]:
    return [
        scale_count(true_count),
        max(0.0, min(1.0, decks_remaining / DECK_SCALE)),
    ]


def feature_count_for(uses_count: bool) -> int:
    return COUNT_FEATURE_COUNT if uses_count else FEATURE_COUNT


def encode_observation(
    observation: Observation, uses_count: bool = False
) -> List[float]:
    features = [
        observation.player_total / 21.0,
        1.0 if observation.is_soft else 0.0,
        observation.dealer_upcard_value / 11.0,
        1.0 if observation.can_double else 0.0,
    ]
    if uses_count:
        features.append(scale_count(observation.true_count))
    return features


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
        self.true_count = kwargs.get("true_count", 0.0)
        self.bet_units = kwargs.get("bet_units", 1)
        self.bet_features: Optional[List[float]] = kwargs.get("bet_features")
        self.bet_action_index = kwargs.get("bet_action_index", 0)
        self.bet_reward = kwargs.get("bet_reward", 0.0)


class BlackjackEnvironment:
    def __init__(self, **kwargs):
        self.table: TrainTable = kwargs.get("table")
        self.policies: Dict[str, object] = kwargs.get("policies", {})
        self.bet_policies: Dict[str, object] = kwargs.get("bet_policies", {})
        self.agent_kinds: Dict[str, str] = kwargs.get("agent_kinds", {})
        self.reward_scale = kwargs.get("reward_scale", FINAL_BET_SCALE)

    def play_reward(self, episode: Episode, outcome) -> float:
        if self.reward_scale == INITIAL_BET_SCALE:
            return outcome.reward / (episode.bet_units * self.table.minimum_bet)
        return outcome.reward / outcome.bet

    def place_bets(self) -> Dict[str, Tuple[List[float], int]]:
        bet_states: Dict[str, Tuple[List[float], int]] = {}
        if not self.bet_policies:
            return bet_states
        features = encode_bet_state(
            self.table.true_count(), self.table.decks_remaining()
        )
        for player in self.table.get_players():
            bet_policy = self.bet_policies.get(player.player_id)
            if bet_policy is None:
                continue
            action_index = bet_policy.select_action(features)
            player.bet_units = BET_UNITS[action_index]
            bet_states[player.player_id] = (features, action_index)
        return bet_states

    def handle_pending_turn(
        self, pending_turn: PendingTurn, episodes: Dict[str, Episode]
    ) -> None:
        policy = self.policies[pending_turn.player_id]
        features = policy.encode(pending_turn.observation)
        legal_action_indices = encode_legal_actions(pending_turn.legal_actions)
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

    def wagered_units(self, player) -> int:
        return self.table.active_bets[player.player_id] // max(player.base_bet, 1)

    def open_episodes(
        self, bet_states: Dict[str, Tuple[List[float], int]]
    ) -> Dict[str, Episode]:
        episodes: Dict[str, Episode] = {}
        for player in self.table.turn_order:
            bet_state = bet_states.get(player.player_id)
            units = self.wagered_units(player)
            episodes[player.player_id] = Episode(
                player_id=player.player_id,
                agent_kind=self.agent_kinds.get(player.player_id, ""),
                steps=[],
                true_count=player.true_count,
                bet_units=units,
                bet_features=None if bet_state is None else bet_state[0],
                bet_action_index=bet_action_for_units(units),
            )
        return episodes

    def play_hand(self) -> List[Episode]:
        bet_states = self.place_bets()
        self.table.begin_hand()
        episodes = self.open_episodes(bet_states)

        while True:
            pending_turn: Optional[PendingTurn] = self.table.get_pending_turn()
            if pending_turn is None:
                break
            self.handle_pending_turn(pending_turn, episodes)

        outcomes = self.table.settle_hand()
        for player_id, outcome in outcomes.items():
            episode = episodes[player_id]
            episode.reward = self.play_reward(episode, outcome)
            episode.bet_reward = outcome.reward / max(self.table.minimum_bet, 1)
            episode.outcome = outcome
        return list(episodes.values())
