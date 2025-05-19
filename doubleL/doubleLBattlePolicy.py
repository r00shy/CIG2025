from itertools import product
from math import prod
from random import sample

from numpy import argmax
from numpy.random import choice

from vgc2.agent import BattlePolicy
from vgc2.battle_engine import State, BattleCommand, calculate_damage, BattleRuleParam, BattlingTeam, BattlingPokemon, \
    BattlingMove, TeamView
from vgc2.util.forward import copy_state, forward
from vgc2.util.rng import ZERO_RNG, ONE_RNG
from typing import Optional

class DoubleLBattlePolicy(BattlePolicy):
    """
    Policy that selects moves and switches randomly. Tailored for single and double battles.
    """

    def __init__(self,
                 switch_prob: float = .15):
        self.switch_prob = switch_prob

    def decision(self,
                 state: State,
                opp_view: Optional[TeamView] = None) -> list[BattleCommand]:
        team = state.sides[0].team
        n_switches = len(team.reserve)
        n_targets = len(state.sides[1].team.active)
        cmds: list[BattleCommand] = []
        for pkm in team.active:
            n_moves = len(pkm.battling_moves)
            switch_prob = 0 if n_switches == 0 else self.switch_prob
            action = choice(n_moves + 1, p=[switch_prob] + [(1. - switch_prob) / n_moves] * n_moves) - 1
            if action >= 0:
                target = choice(n_targets, p=[1 / n_targets] * n_targets)
            else:
                target = choice(n_switches, p=[1 / n_switches] * n_switches)
            cmds += [(action, target)]
        return cmds

