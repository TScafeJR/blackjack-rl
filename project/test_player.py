import unittest

from .game import DecisionInfo, Observation
from .player import Player, PlayerDecision, PlayerType


def build_observation(total, upcard, **kwargs):
    return Observation(
        player_total=total,
        dealer_upcard_value=upcard,
        is_soft=kwargs.get("is_soft", False),
        can_double=kwargs.get("can_double", True),
        money=1000,
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


if __name__ == "__main__":
    unittest.main()
