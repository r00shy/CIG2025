from itertools import product
from math import prod
from random import sample
from game_env import PokemonBattleEnv
from stable_baselines3 import PPO

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
                opp_view: Optional[TeamView] = None) -> list[BattleCommand]: # battlecommand: tuple  (action, target)
        env = PokemonBattleEnv()
        model = PPO.load("ppo_pokemon")

        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info =  env.step(action)
        return action
            