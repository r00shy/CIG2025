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
        commands = []
        my_team = state.sides[0].team
        opponent_team = state.sides[1].team
        field = state.field  # Initialize the field; adjust as per your game's context

        for my_pokemon in my_team.active:
            best_move_index = None
            best_target_index = None
            max_damage = -1

            for move_index, move in enumerate(my_pokemon.battling_moves):
                for target_index, opponent_pokemon in enumerate(opponent_team.active):
                    damage = calculate_damage(
                        params= BattleRuleParam(),
                        attacking_side=0,
                        move=move.constants,
                        state= state,
                        attacker= my_pokemon,
                        defender= opponent_pokemon
                    )
                    if (damage >max_damage):
                        max_damage = damage
                        best_move_index = move_index
                        best_target_index = target_index

            if best_move_index is not None and best_target_index is not None:
                commands.append((best_move_index, best_target_index))

        return commands

