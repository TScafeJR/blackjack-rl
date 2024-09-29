import project

TABLE_COUNT = 3
ROUNDS = 250


def build_table() -> project.Table:
    return (
        project.Table(num_decks=4, minimum_bet=10)
        .add_dealer(project.Dealer())
        .add_player(project.Player(starting_money=5000))
        .add_player(
            project.Player(
                starting_money=5000, player_type=project.PlayerType.AGGRESSIVE
            )
        )
        .add_player(
            project.Player(starting_money=5000, player_type=project.PlayerType.RANDOM)
        )
        .add_player(
            project.Player(
                starting_money=5000, player_type=project.PlayerType.APPREHENSIVE
            )
        )
    )


if __name__ == "__main__":
    game_casino = project.Casino()
    for _ in range(TABLE_COUNT):
        game_casino.add_table(build_table())
    game_casino.give_table_money(100000)

    for _ in range(ROUNDS):
        if not game_casino.can_play_hand():
            break
        game_casino.play_round()

    for table_number, game_table in enumerate(game_casino.get_tables()):
        print(f"--- table {table_number} ---")
        game_table.view_stats()
