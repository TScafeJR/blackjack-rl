import argparse
import json
import os
import random
import webbrowser
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Dict, List, Optional

from project import Dealer, Player, PlayerType, TrainTable
from train.base_learner import build_network
from train.config import HEURISTIC_KINDS, LEARNING_KINDS, parse_agents
from train.environment import ACTION_DECISIONS, encode_observation
from train.policies import BasePolicy, HeuristicPolicy, NetworkPolicy

from .run_store import latest_run_path

ACTION_NAMES = ["hit", "stay", "double"]
BOARD_PAGE = os.path.join(os.path.dirname(__file__), "board.html")


class BoardSession:
    def __init__(self, **kwargs):
        self.run_path = kwargs.get("run_path", "")
        self.agents_spec = kwargs.get("agents", "dqn=1")
        self.epsilon = kwargs.get("epsilon", 0.0)
        self.rng = random.Random(kwargs.get("seed"))
        self.hidden_sizes = kwargs.get("hidden_sizes", [32, 32])
        self.table = TrainTable(
            num_decks=kwargs.get("num_decks", 4),
            minimum_bet=kwargs.get("minimum_bet", 10),
            rebuy=True,
        )
        self.table.add_dealer(Dealer())
        self.policies: Dict[str, BasePolicy] = {}
        self.network_policies: Dict[str, NetworkPolicy] = {}
        self.agent_kinds: Dict[str, str] = {}
        self.hands_played = 0
        self.history: List[dict] = []
        self.history_limit = kwargs.get("history_limit", 200)
        self.handle_seats(kwargs.get("starting_money", 1000))

    def load_network_policy(self, kind: str) -> NetworkPolicy:
        if kind in self.network_policies:
            return self.network_policies[kind]
        weights_path = os.path.join(self.run_path, f"weights_{kind}.json")
        if not os.path.exists(weights_path):
            raise Exception(f"No {kind} weights found at {weights_path}")
        network = build_network(self.hidden_sizes, self.rng)
        network.set_training(False)
        network.load(weights_path)
        policy = NetworkPolicy(network=network, epsilon=self.epsilon, rng=self.rng)
        self.network_policies[kind] = policy
        return policy

    def handle_seats(self, starting_money: int) -> None:
        for kind, count in parse_agents(self.agents_spec):
            for _ in range(count):
                player = Player(
                    starting_money=starting_money, player_type=PlayerType.RANDOM
                )
                self.table.add_player(player)
                self.agent_kinds[player.player_id] = kind
                if kind in LEARNING_KINDS:
                    self.policies[player.player_id] = self.load_network_policy(kind)
                else:
                    self.policies[player.player_id] = HeuristicPolicy(
                        player_type=HEURISTIC_KINDS[kind], rng=self.rng
                    )

    def snapshot_player(self, player: Player) -> dict:
        return {
            "player_id": player.player_id,
            "kind": self.agent_kinds[player.player_id],
            "cards": player.see_hand(),
            "total": player.get_hand_value(),
            "is_soft": player.has_soft_hand(),
            "money": player.get_money(),
        }

    def q_values_for(self, player_id: str, features: List[float]) -> Optional[dict]:
        policy = self.policies[player_id]
        if not isinstance(policy, NetworkPolicy):
            return None
        q_values = policy.network.forward([features])[0]
        return dict(zip(ACTION_NAMES, q_values))

    def build_decision_event(self, pending_turn) -> dict:
        features = encode_observation(pending_turn.observation)
        action_index = self.policies[pending_turn.player_id].select_action(
            pending_turn, features
        )
        return {
            "type": "decision",
            "player_id": pending_turn.player_id,
            "action": ACTION_NAMES[action_index],
            "action_index": action_index,
            "legal_actions": [
                ACTION_NAMES[ACTION_DECISIONS.index(decision)]
                for decision in pending_turn.legal_actions
            ],
            "q_values": self.q_values_for(pending_turn.player_id, features),
        }

    def record_history(
        self, final_players, outcomes, actions_by_player, dealer_event
    ) -> None:
        self.history.append(
            {
                "hand_number": self.hands_played,
                "dealer_cards": dealer_event["cards"],
                "dealer_total": dealer_event["total"],
                "seats": [
                    {
                        "player_id": snapshot["player_id"],
                        "kind": snapshot["kind"],
                        "cards": snapshot["cards"],
                        "total": snapshot["total"],
                        "actions": actions_by_player.get(snapshot["player_id"], []),
                        "result": outcomes[snapshot["player_id"]].result.value,
                        "reward": outcomes[snapshot["player_id"]].reward,
                        "money": outcomes[snapshot["player_id"]].money_after,
                    }
                    for snapshot in final_players
                ],
            }
        )
        if len(self.history) > self.history_limit:
            self.history.pop(0)

    def play_hand_events(self) -> List[dict]:
        events: List[dict] = []
        self.table.begin_hand()
        self.hands_played += 1
        dealer = self.table.get_dealer()
        actions_by_player: Dict[str, List[str]] = {}
        events.append(
            {
                "type": "deal",
                "hand_number": self.hands_played,
                "players": [
                    self.snapshot_player(player) for player in self.table.turn_order
                ],
                "dealer_upcard": dealer.preview_card().to_string(),
                "bets": dict(self.table.active_bets),
            }
        )

        while True:
            pending_turn = self.table.get_pending_turn()
            if pending_turn is None:
                break
            event = self.build_decision_event(pending_turn)
            player = self.table.turn_order[self.table.turn_index]
            self.table.apply_decision(
                pending_turn.player_id, ACTION_DECISIONS[event["action_index"]]
            )
            event["player"] = self.snapshot_player(player)
            event["bet"] = self.table.active_bets[pending_turn.player_id]
            actions_by_player.setdefault(pending_turn.player_id, []).append(
                event["action"]
            )
            events.append(event)

        self.table.play_dealer_hand()
        dealer_event = {
            "type": "dealer",
            "cards": dealer.see_hand(),
            "total": dealer.get_hand_value(),
            "is_bust": dealer.has_bust_hand(),
        }
        events.append(dealer_event)

        final_players = [
            self.snapshot_player(player) for player in self.table.turn_order
        ]
        outcomes = self.table.settle_hand()
        self.record_history(final_players, outcomes, actions_by_player, dealer_event)
        events.append(
            {
                "type": "settle",
                "results": [
                    {
                        "player_id": player_id,
                        "kind": self.agent_kinds[player_id],
                        "result": outcome.result.value,
                        "reward": outcome.reward,
                        "bet": outcome.bet,
                        "money": outcome.money_after,
                    }
                    for player_id, outcome in outcomes.items()
                ],
                "players": final_players,
            }
        )
        return events

    def describe(self) -> dict:
        return {
            "run_path": self.run_path,
            "agents": self.agents_spec,
            "epsilon": self.epsilon,
            "hands_played": self.hands_played,
            "players": [self.snapshot_player(player) for player in self.table.players],
            "kinds": self.agent_kinds,
        }


class BoardRequestHandler(BaseHTTPRequestHandler):
    session: BoardSession = None

    def send_payload(self, payload: bytes, content_type: str) -> None:
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def send_json(self, data: dict | list) -> None:
        self.send_payload(json.dumps(data).encode("utf-8"), "application/json")

    def do_GET(self) -> None:
        path = self.path.split("?")[0]
        if path == "/":
            with open(BOARD_PAGE, "rb") as page_file:
                self.send_payload(page_file.read(), "text/html; charset=utf-8")
            return
        if path == "/api/hand":
            self.send_json(self.session.play_hand_events())
            return
        if path == "/api/state":
            self.send_json(self.session.describe())
            return
        if path == "/api/history":
            self.send_json(self.session.history)
            return
        self.send_response(404)
        self.end_headers()

    def log_message(self, *args) -> None:
        return


def serve(session: BoardSession, port: int, open_browser: bool) -> None:
    BoardRequestHandler.session = session
    server = HTTPServer(("127.0.0.1", port), BoardRequestHandler)
    url = f"http://127.0.0.1:{port}"
    print(f"board: {url}")
    if open_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        server.server_close()


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Watch trained agents play blackjack")
    parser.add_argument("run_path", nargs="?", default=None)
    parser.add_argument("--agents", default="dqn=1")
    parser.add_argument("--port", type=int, default=8181)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--epsilon", type=float, default=0.0)
    parser.add_argument("--starting-money", type=int, default=1000)
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args(argv)

    run_path = args.run_path if args.run_path is not None else latest_run_path()
    print(f"run: {run_path}")
    session = BoardSession(
        run_path=run_path,
        agents=args.agents,
        seed=args.seed,
        epsilon=args.epsilon,
        starting_money=args.starting_money,
    )
    serve(session, args.port, not args.no_browser)


if __name__ == "__main__":
    main()
