from numpy.random import choice, multinomial

from typing import List
from vgc2.battle_engine.modifiers import Category, Status, Hazard, Weather, Terrain, Type
from vgc2.agent import TeamBuildPolicy, TeamBuildCommand
from vgc2.battle_engine.modifiers import Nature, Type
from vgc2.battle_engine.move import Move
from vgc2.meta import Meta, Roster


class DoubleLTeamBuildPolicy(TeamBuildPolicy):
    """
    random team builder.
    """

    def decision(self,
                roster: Roster,
                meta: Meta | None,
                max_team_size: int,
                max_pkm_moves: int,
                n_active: int) -> TeamBuildCommand:

        ivs = (31,) * 6
        top_indices = select_top_offensive_pokemon(roster)
        cmds: TeamBuildCommand = []

        min_power = 0  # initial power threshold
        required_count = 4

        # Make a copy to safely modify
        valid_indices = top_indices.copy()

        while len(valid_indices) > required_count:
            filtered = []

            for i in valid_indices:
                stats = {
                    'ATK': roster[i].base_stats[1],
                    'S_ATK': roster[i].base_stats[2],
                    'SPD': roster[i].base_stats[5]
                }

                role = infer_pokemon_role(stats)

                if has_two_strong_offensive_moves(roster[i].moves, role, min_power):
                    filtered.append(i)

            # Stop if filtering would drop below required_count
            if len(filtered) < required_count:
                break

            valid_indices = filtered
            min_power += 5  # increase requirement for next round

        top4_indices = valid_indices[:required_count]

        for i in top4_indices:
            stats = {
                'ATK': roster[i].base_stats[1],
                'S_ATK': roster[i].base_stats[2],
                'SPD': roster[i].base_stats[5]
            }

            role = infer_pokemon_role(stats)
            evs = tuple(multinomial(510, [1 / 6] * 6, size=1)[0])
            nature = choose_optimal_nature_role_based(stats, role)
            moves = choose_optimal_moveset(roster[i].types, roster[i].moves, max_pkm_moves, role) 
            
            cmds.append((i, evs, ivs, nature, moves))

        return cmds
    
def select_top_offensive_pokemon(roster: Roster, top_n: int = 30) -> list[int]:
    # Build a list of (index, offensive stat) pairs
    indexed_stats = [
        (i, max(pokemon.base_stats[1], pokemon.base_stats[2]))  # ATK = index 1, S_ATK = index 2
        for i, pokemon in enumerate(roster)
    ]

    # Sort by offensive stat descending
    indexed_stats.sort(key=lambda x: x[1], reverse=True)

    # Extract the top N indices
    top_indices = [i for i, _ in indexed_stats[:top_n]]

    return top_indices

def has_two_strong_offensive_moves(moves: List[Move], role: Category, min_power: int) -> bool:
    strong_moves = set()

    cat = role_to_category(role)
    for move in moves:
        if move.category == cat and move.base_power >= min_power:
            strong_moves.add(move.pkm_type)

        if len(strong_moves) >= 2:
            return True

    return False

def role_to_category(role: str) -> Category:
    role = role.lower()
    if "physical" in role:
        return Category.PHYSICAL
    else:
        return Category.SPECIAL

def choose_optimal_nature_role_based(stats: dict, role: str) -> Nature:
    atk = stats.get('ATK', 0)
    s_atk = stats.get('S_ATK', 0)
    spd = stats.get('SPD', 0)

    if spd > 100:
        # Fast Pokémon get speed-boosting nature
        if role == 'physical_attacker':
            return Nature.JOLLY  # +SPD, -S_ATK
        elif role == 'special_attacker':
            return Nature.TIMID  # +SPD, -ATK
    else:
        # Otherwise, boost the attack stat and reduce the other
        if role == 'physical_attacker':
            return Nature.ADAMANT  # +ATK, -S_ATK
        elif role == 'special_attacker':
            return Nature.MODEST  # +S_ATK, -ATK

    return Nature.HARDY  # fallback neutral
    
def infer_pokemon_role(stats: dict) -> str:
    atk = stats.get('ATK', 0)
    s_atk = stats.get('S_ATK', 0)
    spd = stats.get('SPD', 0)

    # Speed is important: if one attacker type is clearly stronger and the Pokémon is fast, prefer that
    if atk >= s_atk:
        return 'physical_attacker'
    else:
        return 'special_attacker'

def choose_optimal_moveset(pokemonTypes: list[Type], moves: list[Move], max_pkm_moves, role) -> list[int]:
    # TODO: Implement a more sophisticated moveset selection based on role
    list_of_moves = []
    for i in range(min(len(moves), max_pkm_moves)):
        if moves[i].pkm_type in pokemonTypes:
            # Check if the move is STAB (Same Type Attack Bonus)
            list_of_moves.append(i)
    if len(list_of_moves) < max_pkm_moves:
        # If not enough STAB moves, fill with other moves
        for i in range(len(moves)):
            if i not in list_of_moves:
                list_of_moves.append(i)
                if len(list_of_moves) == max_pkm_moves:
                    break
    return list_of_moves
