from environment import Environment
from project import Casino, Player, Table, Dealer, PlayerType
from agent import Agent

if __name__ == '__main__':
    casino = Casino()
    NUM_STEPS = 1000

    starting_money = 5000

    dealer = Dealer()
    game_table = Table(4, 10).add_dealer(dealer)

    random_player = Player(starting_money, PlayerType.RANDOM)
    aggressive_player = Player(starting_money, PlayerType.AGGRESSIVE)
    conservative_player = Player(starting_money, PlayerType.APPREHENSIVE)
    noob_player = Player(starting_money, PlayerType.NOOB)

    # Create players and agents
    players = [
        random_player,
        aggressive_player,
        conservative_player,
        noob_player
    ]

    agents = [Agent(player) for player in players]

    # Add players to the table
    for player in players:
        game_table.add_player(player)

    casino.add_table(game_table).give_table_money(50000)

    environment = Environment(casino)

    while casino.can_play_hand():
        observations = environment.reset()
        terminal = False

        while not terminal:

