import project

if __name__ == '__main__':
    game_casino = project.Casino()
    game_dealer = project.Dealer()
    player1 = project.Player(5000)

    game_table = project.Table(4, 10)\
        .add_dealer(game_dealer)\
        .add_player(player1)

    game_casino\
        .add_table(game_table)\
        .give_table_money(10000)

    while player1.get_money() > 0 and game_casino.table.get_money() > 0 and\
            game_casino.table.get_stats()['hands_played'] < 1000:
        game_casino.table.play_hand()

    game_casino.table.view_stats()
