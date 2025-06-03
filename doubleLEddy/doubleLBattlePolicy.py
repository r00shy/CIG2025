from itertools import product
from math import prod
from random import sample

from numpy import argmax
from numpy.random import choice

from vgc2.agent import BattlePolicy
from vgc2.battle_engine import State, BattleCommand, calculate_damage, BattleRuleParam, BattlingTeam, BattlingPokemon, \
    BattlingMove, TeamView
from vgc2.battle_engine.damage_calculator import calculate_boosted_stats
from vgc2.battle_engine.modifiers import Stat
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

        # Variables to store best and alternative moves/targets per Pokémon
        best_move_indices = []
        best_target_indices = []
        max_damages = []
        alt_move_indices = []
        alt_target_indices = []
        alt_max_damages = []

        # Loop exactly as you wrote it (assuming two active Pokémon)
        for my_pokemon in my_team.active:
            best_move_index = None
            best_target_index = None
            max_damage = -1
            alternativ_move_index = None
            alternativ_target_index = None
            alternativ_max_damage = -1

            for move_index, move in enumerate(my_pokemon.battling_moves):
                for target_index, opponent_pokemon in enumerate(opponent_team.active):
                    damage = calculate_damage(
                        params=BattleRuleParam(),
                        attacking_side=0,
                        move=move.constants,
                        state=state,
                        attacker=my_pokemon,
                        defender=opponent_pokemon
                    )
                    if damage > max_damage:
                        if target_index != best_target_index:
                            alternativ_move_index = best_move_index
                            alternativ_target_index = best_target_index
                            alternativ_max_damage = max_damage

                        max_damage = damage
                        best_move_index = move_index
                        best_target_index = target_index
                    elif damage > alternativ_max_damage and target_index != best_target_index:
                        alternativ_max_damage = damage
                        alternativ_move_index = move_index
                        alternativ_target_index = target_index

            best_move_indices.append(best_move_index)
            best_target_indices.append(best_target_index)
            max_damages.append(max_damage)
            alt_move_indices.append(alternativ_move_index)
            alt_target_indices.append(alternativ_target_index)
            alt_max_damages.append(alternativ_max_damage)

        # If you have only one Pokémon, just add one command
        if len(my_team.active) == 1:
            commands.append((best_move_indices[0], best_target_indices[0]))
            return commands

        # If opponent only has one Pokémon and your pokemon have different targets, both use the best move
        if len(opponent_team.active) == 1 or best_target_indices[0] != best_move_indices[1]:
            commands.append((best_move_indices[0], best_target_indices[0]))
            commands.append((best_move_indices[1], best_target_indices[1]))
            return commands

        # If the best moves target the same pokemon, lets apply some logic based on our pokemon speed:
        my_pokemon_0_speed = BattleRuleParam().BOOST_MULTIPLIER_LOOKUP[my_team.active[0].boosts[Stat.SPEED]]* my_team.active[0].constants.stats[Stat.SPEED]
        my_pokemon_1_speed = BattleRuleParam().BOOST_MULTIPLIER_LOOKUP[my_team.active[1].boosts[Stat.SPEED]]* my_team.active[1].constants.stats[Stat.SPEED]

        fasterIndex = 0
        slowerIndex = 1

        if my_pokemon_0_speed < my_pokemon_1_speed:
            fasterIndex = 1
            slowerIndex = 0
        
        # Now apply the lethal check and select final commands
        # Check if faster Pokémon has a lethal move
        if (max_damages[fasterIndex] >= opponent_team.active[best_target_indices[fasterIndex]].hp):
            if (fasterIndex == 0):
                commands.append((best_move_indices[fasterIndex], best_target_indices[fasterIndex]))
                commands.append((alt_move_indices[slowerIndex], alt_target_indices[slowerIndex]))
            else:
                commands.append((best_move_indices[slowerIndex], best_target_indices[slowerIndex]))
                commands.append((alt_move_indices[fasterIndex], alt_target_indices[fasterIndex]))
            return commands
        
        if (max_damages[slowerIndex] >= opponent_team.active[best_target_indices[slowerIndex]].hp):
                if (slowerIndex == 0):
                    commands.append((best_move_indices[slowerIndex], best_target_indices[slowerIndex]))
                    commands.append((alt_move_indices[fasterIndex], alt_target_indices[fasterIndex]))
                else:
                    commands.append((best_move_indices[fasterIndex], best_target_indices[fasterIndex]))
                    commands.append((alt_move_indices[slowerIndex], alt_target_indices[slowerIndex]))
                return commands
        
        # Check then if slower Pokémon has a lethal move
        if (max_damages[fasterIndex] >= opponent_team.active[best_target_indices[fasterIndex]].hp):
                if (slowerIndex == 0):
                    commands.append((best_move_indices[slowerIndex], best_target_indices[slowerIndex]))
                    commands.append((alt_move_indices[fasterIndex], alt_target_indices[fasterIndex]))
                else:
                    commands.append((best_move_indices[fasterIndex], best_target_indices[fasterIndex]))
                    commands.append((alt_move_indices[slowerIndex], alt_target_indices[slowerIndex]))
                return commands
                    

        # Otherwise: both use their best moves
        if (slowerIndex == 0):
            commands.append((best_move_indices[slowerIndex], best_target_indices[slowerIndex]))
            commands.append((best_move_indices[fasterIndex], best_target_indices[fasterIndex]))
        else:
            commands.append((best_move_indices[fasterIndex], best_target_indices[fasterIndex]))
            commands.append((best_move_indices[slowerIndex], best_target_indices[slowerIndex]))
        return commands

