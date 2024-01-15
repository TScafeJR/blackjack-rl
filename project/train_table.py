from .base_table import BaseTable


class TrainTable(BaseTable):
    def __init__(self, num_decks: int, minimum_bet: int):
        super().__init__(num_decks, minimum_bet)

    @override
    def play_hand(self):
