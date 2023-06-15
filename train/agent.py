from environment import Environment
from project import Player
from typing import List


class Agent:
    internal_player: Player
    total_reward: int = 0

    def __init__(self, player: Player):
        super().__init__()
        self.internal_player = player

    def get_reward(self, action: List[Player]) -> int:
        for player in action:
            if player.player_id == self.internal_player.player_id:
                return player.get_last_hand_res()

    def get_hands_played(self) -> int:
        return self.internal_player.get_hands_played()

    def type_as_str(self):
        return self.internal_player.type_as_str()

    def step(self, ob: Environment):
        self.total_reward += self.get_reward(ob.action())
