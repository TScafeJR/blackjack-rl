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

    casino.add_table(game_table).give_table_money(10000)

    environment = Environment(casino)

    while NUM_STEPS > 0:
        NUM_STEPS -= 1

        # Get the players from the environment
        casino_players = environment.get_agents()

        # Stop if all players run out of money
        if all(player.get_money() <= 0 for player in casino_players):
            break

        # All players make their decision and table plays a hand
        results_of_hands = environment.action()

        # Let each agent get reward
        for agent in agents:
            agent.total_reward += agent.get_reward(results_of_hands)

    # Print hands played by each agent
    for i, agent in enumerate(agents):
        print(f"Player {agent.type_as_str()} hands played: {agent.get_hands_played()}, total reward: {agent.total_reward}")

