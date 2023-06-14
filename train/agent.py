from environment import Environment
from project import Player


class Agent:
    internal_player: Player
    total_reward: int = 0

    def __init__(self, player: Player):
        super().__init__()
        self.internal_player = player

    def step(self, ob: Environment):
        curr_obs = ob.get_observation()
        print(curr_obs)
        curr_action = ob.get_actions()
        print(curr_action)
        curr_reward = ob.action()
        self.total_reward += curr_reward[0]
