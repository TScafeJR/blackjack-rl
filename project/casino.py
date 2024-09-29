from __future__ import annotations

from typing import List

from .table import Table


class Casino:
    def __init__(self, **kwargs):
        self.money = kwargs.get("starting_money", 1000000)
        self.tables: List[Table] = []

    def add_table(self, new_table: Table) -> Casino:
        self.tables.append(new_table)
        return self

    def get_tables(self) -> List[Table]:
        return self.tables

    def get_money(self) -> int:
        return self.money

    def give_table_money(self, amount: int) -> Casino:
        if len(self.tables) == 0:
            raise Exception("No tables are defined")

        for table in self.tables:
            table.receive_money(amount)
            self.money -= amount
        return self

    def can_play_hand(self) -> bool:
        for table in self.tables:
            if table.can_play_hand():
                return True
        return False

    def play_round(self) -> None:
        for table in self.tables:
            if table.can_play_hand():
                table.play_hand()
