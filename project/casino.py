from __future__ import annotations
from .table import Table


class Casino:
    def __init__(self):
        self.money = 100000
        self.table = None
        self.dealer = None

    def add_table(self, new_table: Table) -> Casino:
        self.table = new_table
        return self

    def get_money(self) -> int:
        return self.money

    def give_table_money(self, amount: int) -> Casino:
        self.table.receive_money(amount)
        self.money -= amount
        return self
