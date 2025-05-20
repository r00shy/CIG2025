from numpy.random import choice, multinomial
from collections import defaultdict
from vgc2.agent import TeamBuildPolicy, TeamBuildCommand
from vgc2.battle_engine.modifiers import Nature, Type
from vgc2.battle_engine.move import Move
from vgc2.battle_engine.pokemon import PokemonSpecies
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
        
        ivs = (31,) * 6 # ivs = (31, 31, 31, 31, 31, 31) = perfect individual values
        ids = choose_optimal_pokemon(roster, max_team_size) # choice(len(roster), max_team_size, False) # TODO: Choose with strategy
        cmds: TeamBuildCommand = []
        for i in range(len(ids)):
            # Choose the role of the pokemon based on the stats
            statDict = {
                'ATK': roster[i].base_stats[1],
                'S_ATK': roster[i].base_stats[2],
                'DEF': roster[i].base_stats[3],
                'S_DEF': roster[i].base_stats[4],
                'SPD': roster[i].base_stats[5]
            }
            role = infer_pokemon_role(statDict) 
            # print(role)
            # print(roster[i].base_stats)
            # print(roster[i].moves[0])
            n_moves = len(roster[i].moves) # no need to change; number of moves (probably 4)

            moves = choose_optimal_moveset(roster[i].types, roster[i].moves, max_pkm_moves, role) 
            # moves = list(choice(n_moves, min(max_pkm_moves, n_moves), False)) # DEFAULT VERSION
            evs = tuple(multinomial(510, [1 / 6] * 6, size=1)[0]) # this is in alignment with the rules; Each of the 6 stats has an equal probability of receiving EVs; overall less than or equal to 510
            
            nature = choose_optimal_nature_role_based(statDict, role)
            cmds += [(i, evs, ivs, nature, moves)]
        return cmds
    

def choose_optimal_nature_role_based(stats: dict, role: str) -> Nature:
    nature_chart = {
        ('ATK', 'S_ATK'): Nature.ADAMANT,
        ('ATK', 'S_DEF'): Nature.NAUGHTY,
        ('ATK', 'DEF'): Nature.LONELY,
        ('ATK', 'SPD'): Nature.BRAVE,
        ('S_ATK', 'ATK'): Nature.MODEST,
        ('S_ATK', 'DEF'): Nature.MILD,
        ('S_ATK', 'S_DEF'): Nature.RASH,
        ('S_ATK', 'SPD'): Nature.QUIET,
        ('SPD', 'ATK'): Nature.TIMID,
        ('SPD', 'DEF'): Nature.HASTY,
        ('SPD', 'S_ATK'): Nature.JOLLY,
        ('SPD', 'S_DEF'): Nature.NAIVE,
        ('DEF', 'ATK'): Nature.BOLD,
        ('DEF', 'S_ATK'): Nature.IMPISH,
        ('DEF', 'SPD'): Nature.RELAXED,
        ('DEF', 'S_DEF'): Nature.LAX,
        ('S_DEF', 'ATK'): Nature.CALM,
        ('S_DEF', 'S_ATK'): Nature.CAREFUL,
        ('S_DEF', 'DEF'): Nature.GENTLE,
        ('S_DEF', 'SPD'): Nature.SASSY
    }

    role_boost_map = {
        'physical_attacker': 'ATK',
        'special_attacker': 'S_ATK',
        'speedster': 'SPD',
        'physical_wall': 'DEF',
        'special_wall': 'S_DEF',
        'trick_room': 'ATK'  # Trick Room attackers often prioritize ATK, don't mind low SPD
    }

    safe_reduce_map = {
        'physical_attacker': ['S_ATK', 'S_DEF'],
        'special_attacker': ['ATK', 'DEF'],
        'speedster': ['S_ATK', 'DEF', 'S_DEF'],
        'physical_wall': ['S_ATK', 'SPD'],
        'special_wall': ['ATK', 'SPD'],
        'trick_room': ['SPD', 'S_ATK', 'S_DEF']
    }

    boost = role_boost_map.get(role)
    safe_to_reduce = [s for s in safe_reduce_map.get(role, []) if s != boost]
    
    # Choose the stat to reduce (lowest value among safe options)
    reduce = min(safe_to_reduce, key=lambda s: stats.get(s, float('inf')))

    nature = nature_chart.get((boost, reduce), 'Hardy')

    return nature
    
def infer_pokemon_role(stats: dict) -> str:
    atk = stats.get('ATK', 0)
    s_atk = stats.get('S_ATK', 0)
    def_ = stats.get('DEF', 0)
    s_def = stats.get('S_DEF', 0)
    spd = stats.get('SPD', 0)

    # Trick Room candidate: low speed and good offense
    if spd < 60 and (atk > 90 or s_atk > 90):
        return 'trick_room'

    # Speedster: very fast and offensive
    if spd > 100 and (atk > 85 or s_atk > 85):
        return 'speedster'

    # Physical Attacker
    if atk >= max(s_atk, def_, s_def, spd):
        return 'physical_attacker'

    # Special Attacker
    if s_atk >= max(atk, def_, s_def, spd):
        return 'special_attacker'

    # Physical Wall
    if def_ >= max(atk, s_atk, s_def, spd):
        return 'physical_wall'

    # Special Wall
    if s_def >= max(atk, s_atk, def_, spd):
        return 'special_wall'

    # Fallback default
    return 'physical_attacker'

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

def choose_optimal_pokemon(roster: list[PokemonSpecies], max_team_size: int) -> list[int]:
    from collections import defaultdict

    # Step 1: Group Pokémon by their type combination
    groups = defaultdict(list)
    for p in roster:
        key = tuple(sorted(p.types))
        groups[key].append(p)

    # Step 2: Define a basic type synergy scoring function
    def type_synergy_score(type_combo: tuple[Type]) -> int:
        """
        Higher score means better synergy.
        - Favors diversity (2 distinct types).
        - Penalizes common weaknesses.
        - Could be extended with resistances and immunities.
        """
        score = 0
        if len(set(type_combo)) == 2:
            score += 10  # bonus for dual types
        # Example bonuses for some known good pairings
        good_synergies = [
            (Type.FAIRY, Type.STEEL), (Type.WATER, Type.GROUND), (Type.DRAGON, Type.FLYING),
            (Type.BUG, Type.STEEL), (Type.DARK, Type.GHOST), (Type.GROUND, Type.FLYING)
        ]
        if type_combo in good_synergies or tuple(reversed(type_combo)) in good_synergies:
            score += 20
        return score

    # Step 3: Compute synergy scores
    scored_type_combos = sorted(
        groups.items(),
        key=lambda item: type_synergy_score(item[0]),
        reverse=True
    )

    # Step 4: From each of the top scoring type combos, choose the Pokémon with highest stats
    best_pokemon = []
    for type_combo, pokemon_list in scored_type_combos:
        top_pokemon = max(pokemon_list, key=lambda p: sum(p.base_stats))
        best_pokemon.append(top_pokemon)
        if len(best_pokemon) >= max_team_size:
            break

    # Step 5: Return the IDs of the selected Pokémon
    return [p.id for p in best_pokemon]
