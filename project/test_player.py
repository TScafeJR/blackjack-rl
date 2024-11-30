import unittest

from .game import DecisionInfo, Observation
from .player import Player, PlayerDecision, PlayerType
from .train_table import TrainTable


def build_observation(total, upcard, **kwargs):
    return Observation(
        player_total=total,
        dealer_upcard_value=upcard,
        is_soft=kwargs.get("is_soft", False),
        can_double=kwargs.get("can_double", True),
        money=1000,
        true_count=kwargs.get("true_count", 0.0),
    )


class TestBasicStrategy(unittest.TestCase):
    def test_hard_low_totals_hit(self):
        for total in [4, 5, 6, 7, 8]:
            decision = Player.decide_basic_strategy(build_observation(total, 6))
            self.assertEqual(decision, PlayerDecision.HIT)

    def test_hard_eleven_doubles(self):
        decision = Player.decide_basic_strategy(build_observation(11, 11))
        self.assertEqual(decision, PlayerDecision.DOUBLE_DOWN)

    def test_hard_eleven_without_double_hits(self):
        decision = Player.decide_basic_strategy(
            build_observation(11, 11, can_double=False)
        )
        self.assertEqual(decision, PlayerDecision.HIT)

    def test_hard_ten_doubles_against_nine_not_ten(self):
        self.assertEqual(
            Player.decide_basic_strategy(build_observation(10, 9)),
            PlayerDecision.DOUBLE_DOWN,
        )
        self.assertEqual(
            Player.decide_basic_strategy(build_observation(10, 10)),
            PlayerDecision.HIT,
        )

    def test_hard_twelve_stands_only_against_four_to_six(self):
        self.assertEqual(
            Player.decide_basic_strategy(build_observation(12, 4)),
            PlayerDecision.STAY,
        )
        self.assertEqual(
            Player.decide_basic_strategy(build_observation(12, 2)),
            PlayerDecision.HIT,
        )

    def test_hard_sixteen_hits_against_ten(self):
        self.assertEqual(
            Player.decide_basic_strategy(build_observation(16, 10)),
            PlayerDecision.HIT,
        )
        self.assertEqual(
            Player.decide_basic_strategy(build_observation(16, 6)),
            PlayerDecision.STAY,
        )

    def test_hard_seventeen_stands(self):
        self.assertEqual(
            Player.decide_basic_strategy(build_observation(17, 11)),
            PlayerDecision.STAY,
        )

    def test_soft_eighteen(self):
        self.assertEqual(
            Player.decide_basic_strategy(build_observation(18, 3, is_soft=True)),
            PlayerDecision.DOUBLE_DOWN,
        )
        self.assertEqual(
            Player.decide_basic_strategy(
                build_observation(18, 3, is_soft=True, can_double=False)
            ),
            PlayerDecision.STAY,
        )
        self.assertEqual(
            Player.decide_basic_strategy(build_observation(18, 9, is_soft=True)),
            PlayerDecision.HIT,
        )

    def test_soft_nineteen_doubles_only_against_six(self):
        self.assertEqual(
            Player.decide_basic_strategy(build_observation(19, 6, is_soft=True)),
            PlayerDecision.DOUBLE_DOWN,
        )
        self.assertEqual(
            Player.decide_basic_strategy(build_observation(19, 5, is_soft=True)),
            PlayerDecision.STAY,
        )

    def test_soft_thirteen_doubles_against_five_and_six(self):
        self.assertEqual(
            Player.decide_basic_strategy(build_observation(13, 5, is_soft=True)),
            PlayerDecision.DOUBLE_DOWN,
        )
        self.assertEqual(
            Player.decide_basic_strategy(build_observation(13, 4, is_soft=True)),
            PlayerDecision.HIT,
        )

    def test_decide_for_type_basic(self):
        decision_info = DecisionInfo(observation=build_observation(11, 6))
        decision = Player.decide_for_type(PlayerType.BASIC, decision_info)
        self.assertEqual(decision, PlayerDecision.DOUBLE_DOWN)


class TestCountingStrategy(unittest.TestCase):
    def test_neutral_count_matches_basic(self):
        for total, upcard in [(16, 10), (12, 2), (13, 2), (11, 6)]:
            self.assertEqual(
                Player.decide_counting(build_observation(total, upcard)),
                Player.decide_basic_strategy(build_observation(total, upcard)),
            )

    def test_sixteen_stands_against_ten_at_positive_count(self):
        self.assertEqual(
            Player.decide_counting(build_observation(16, 10, true_count=1.0)),
            PlayerDecision.STAY,
        )
        self.assertEqual(
            Player.decide_counting(build_observation(16, 10, true_count=0.5)),
            PlayerDecision.HIT,
        )

    def test_fifteen_stands_against_ten_at_high_count(self):
        self.assertEqual(
            Player.decide_counting(build_observation(15, 10, true_count=4.0)),
            PlayerDecision.STAY,
        )
        self.assertEqual(
            Player.decide_counting(build_observation(15, 10, true_count=3.0)),
            PlayerDecision.HIT,
        )

    def test_twelve_stands_against_three_at_plus_two(self):
        self.assertEqual(
            Player.decide_counting(build_observation(12, 3, true_count=2.0)),
            PlayerDecision.STAY,
        )

    def test_thirteen_hits_against_two_at_negative_count(self):
        self.assertEqual(
            Player.decide_counting(build_observation(13, 2, true_count=-1.0)),
            PlayerDecision.HIT,
        )
        self.assertEqual(
            Player.decide_counting(build_observation(13, 2, true_count=0.0)),
            PlayerDecision.STAY,
        )

    def test_ten_doubles_against_ten_at_high_count(self):
        self.assertEqual(
            Player.decide_counting(build_observation(10, 10, true_count=4.0)),
            PlayerDecision.DOUBLE_DOWN,
        )
        self.assertEqual(
            Player.decide_counting(
                build_observation(10, 10, true_count=4.0, can_double=False)
            ),
            PlayerDecision.HIT,
        )

    def test_soft_hands_ignore_deviations(self):
        self.assertEqual(
            Player.decide_counting(
                build_observation(18, 9, is_soft=True, true_count=5.0)
            ),
            Player.decide_basic_strategy(build_observation(18, 9, is_soft=True)),
        )

    def test_decide_for_type_counting(self):
        decision_info = DecisionInfo(
            observation=build_observation(16, 10, true_count=2.0)
        )
        decision = Player.decide_for_type(PlayerType.COUNTING, decision_info)
        self.assertEqual(decision, PlayerDecision.STAY)


class TestCountingBets(unittest.TestCase):
    def setUp(self):
        self.player = Player(starting_money=1000, player_type=PlayerType.COUNTING)

    def test_flat_bet_at_neutral_and_negative_counts(self):
        for true_count in [-3.0, 0.0, 1.9]:
            self.player.true_count = true_count
            self.assertEqual(self.player.get_bet_amount(), 10)

    def test_bet_scales_with_count(self):
        self.player.true_count = 2.4
        self.assertEqual(self.player.get_bet_amount(), 20)
        self.player.true_count = 3.0
        self.assertEqual(self.player.get_bet_amount(), 30)

    def test_bet_caps_at_five_units(self):
        self.player.true_count = 9.0
        self.assertEqual(self.player.get_bet_amount(), 50)

    def test_other_types_bet_flat(self):
        basic_player = Player(starting_money=1000, player_type=PlayerType.BASIC)
        basic_player.true_count = 5.0
        self.assertEqual(basic_player.get_bet_amount(), 10)


class TestBetUnits(unittest.TestCase):
    def test_learner_bet_units_scale_the_wager(self):
        player = Player(starting_money=1000, player_type=PlayerType.RANDOM)
        player.bet_units = 4
        self.assertEqual(player.get_bet_units(), 4)
        self.assertEqual(player.get_bet_amount(), 40)

    def test_base_bet_scales_every_unit(self):
        player = Player(starting_money=1000, base_bet=25)
        player.bet_units = 3
        self.assertEqual(player.get_bet_amount(), 75)

    def test_counting_ignores_externally_set_units(self):
        player = Player(starting_money=1000, player_type=PlayerType.COUNTING)
        player.bet_units = 5
        player.true_count = 0.0
        self.assertEqual(player.get_bet_amount(), 10)

    def test_table_sets_base_bet_from_minimum(self):
        table = TrainTable(num_decks=1, minimum_bet=50, rebuy=False)
        player = Player(starting_money=1000)
        table.add_player(player)
        self.assertEqual(player.base_bet, 50)
        self.assertEqual(player.get_bet_amount(), 50)

    def test_submit_bet_is_capped_by_bankroll(self):
        player = Player(starting_money=25)
        player.bet_units = 5
        self.assertEqual(player.submit_bet(), 25)


if __name__ == "__main__":
    unittest.main()
