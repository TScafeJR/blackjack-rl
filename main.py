import project

if __name__ == '__main__':
    game_casino = project.Casino()
    game_dealer = project.Dealer()
    player1 = project.Player(5000)
    aggressive_player = project.Player(5000, project.PlayerType.AGGRESSIVE)
    random_player = project.Player(5000, project.PlayerType.RANDOM)

    game_table = project.Table(4, 10).add_dealer(game_dealer).add_player(player1).add_player(aggressive_player).add_player(random_player)

    game_casino.add_table(game_table).give_table_money(100000)

    if game_casino.table is not None:
        while (
            game_casino.table.can_play_hand()
            and game_casino.table.get_hands_played() < 1000
        ):
            game_casino.table.play_hand()

        game_casino.table.view_stats()
