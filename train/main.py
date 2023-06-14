from environment import Environment
from project import Casino, Player, Table, Dealer
from agent import Agent

if __name__ == '__main__':
    casino = Casino()

    player = Player(1000)
    dealer = Dealer()

    game_table = Table(4, 10) \
        .add_dealer(dealer) \
        .add_player(player)

    casino \
        .add_table(game_table) \
        .give_table_money(10000)

    obj = Environment(casino)

    step_number = 0
    while not obj.check_is_done():
        step_number += 1
        casino_player = obj.get_agents()[0]
        player_as_agent = Agent(casino_player)
        player_as_agent.step(obj)
