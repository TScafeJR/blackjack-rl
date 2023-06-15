from project import Casino
from typing import List


class Environment:
    def __init__(self, casino: Casino):
        self.remaining_steps = 100
        self.casino = casino

    def get_agents(self):
        return self.casino.table.get_players()

    def get_observation(self) -> List[float]:
        return list(map(lambda p: p.get_money(), self.get_agents()))

    def get_actions(self) -> List[int]:
        return list(map(lambda p: p.get_last_hand_res(), self.get_agents()))

    def check_is_done(self) -> bool:
        return self.remaining_steps == 0

    def action(self):
        return self.casino.table.play_hand()
