from itertools import product
from math import prod
from random import sample
from stable_baselines3 import PPO

from numpy import argmax
from numpy.random import choice

from doubleL.game_env import PokemonBattleEnv
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
    Using stable baselines3 PPO to make decisions in a double battle environment.
    """

    def __init__(self, model_path: str = "models/v1/1200000.zip"):
        self.env = PokemonBattleEnv()  # Only used for observation extraction
        self.model = PPO.load(model_path)

    def decision(self,
                 state: State,
                 opp_view: Optional[TeamView] = None) -> list[tuple[int, int]]:
        # Generate a real-time observation from the current State
        self.env.state = state
        obs = self.env._get_obs()

        # PPO expects batched input, so add batch dimension
        action_masks = self.env.get_action_mask()
        action, _ = self.model.predict(obs, deterministic=True, action_masks=action_masks)

        # Decode action to BattleCommand-like structure
        commands = self.env._decode_action(action)
        return commands
            