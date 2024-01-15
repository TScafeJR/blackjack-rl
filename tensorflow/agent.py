from tensorforce import Agent

class CustomAgent(Agent):
    def __init__(self, player, environment):
        super().__init__()
        self.player = player
        agent = Agent.create(
            agent='dqn',
            environment=environment,
            )

    def act(self, states, deterministic=False):


