from typing import Optional
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from vgc2.battle_engine import State
from vgc2.agent import BattlePolicy
from vgc2.battle_engine import BattleEngine
from vgc2.battle_engine.pokemon import BattlingPokemon
from vgc2.battle_engine.team import Team
from vgc2.battle_engine.view import TeamView, StateView
from vgc2.util.generator import gen_team, _RNG
from vgc2.battle_engine.game_state import get_battle_teams
from vgc2.competition.match import label_teams  # assuming this is a util wrapper

from vgc2.agent.battle import GreedyBattlePolicy


class PokemonBattleEnv(gym.Env):
    def __init__(self, team_size=4, n_moves=10, n_active=2): # why 10 moves: 4 moves * 2 possible targets + 2 switch options = 10
        super().__init__()
        self.team_size = team_size
        self.n_moves = n_moves
        self.n_active = n_active

        self.opponent_policy = GreedyBattlePolicy()

        # Define action space 
        self.action_space = spaces.MultiDiscrete([n_moves, n_moves])  # Needs matching with your actual BattleCommand encoding

        self.observation_space = gym.spaces.Box(-1, 300, shape=(7,), dtype=np.float32)
        
        """ gym.spaces.Dict(
            {
                "ownTeam": gym.spaces.Box(0, 300, shape=(9,4), dtype=np.float32), # 2d vector for 4 Pokémon, each with 9 features
                "opponentTeam": gym.spaces.Box(0, 300, shape=(9,4), dtype=np.float32), 
                "environment": gym.spaces.Box(0, 4, shape=(3,), dtype=np.float32),  # Weather, Terrain, Trickroom, etc.
            }
        ) """

        self.reset()

    def reset(self, seed: Optional[int] = None):
        self.team = gen_team(self.team_size, self.n_moves), gen_team(self.team_size, self.n_moves)
        label_teams(self.team)
        self.team_view = TeamView(self.team[0]), TeamView(self.team[1])
        self.state = State(get_battle_teams(self.team, self.n_active))
        self.state_view = StateView(self.state, 0, self.team_view), StateView(self.state, 1, self.team_view)
        self.engine = BattleEngine(self.state)
        info = {}

        return (self._get_obs(), info)

    def step(self, action):
        # Parse agent action into BattleCommand for side 1
       
        agent_command = self._decode_action(action, self.state_view[0])
        # TODO: possibly help agent by ruling out stupid decisions like switching out a fainted Pokémon
        
        opponent_command = self.opponent_policy.decision(self.state_view[1], self.team_view[0])
        initialOppTotalHp = sum(p.hp for p in self.state.sides[1].team.active)  # Store initial total HP of opponent's active Pokémon
        self.engine.run_turn((opponent_command, agent_command))

        obs = self._get_obs()
        terminated = self.engine.finished()
        reward = self._get_reward(initialOppTotalHp) # TODO: find a better reward function
        print("Action:", agent_command, "Reward:", reward)
        info = {}
        truncated: bool = False  # Not used in this environment, but required by gym API
        return obs, reward, terminated, truncated, info

    def _get_obs(self):
        # TODO: Convert StateView for side 1 into a flat vector or custom structure
        ownActive = self.state.sides[0].team.active
        ownReserve = self.state.sides[0].team.reserve
        """ oppActive = self.state.sides[1].team.active
        oppReserve = self.state.sides[1].team.reserve
        own = [self.encode_pkm(p) for p in ownActive + ownReserve]
        opp = [self.encode_pkm(p) for p in oppActive + oppReserve]

        own = np.array(own, dtype=np.float32)
        opp = np.array(opp, dtype=np.float32) """

        # global feats:
        env_feats = []
        # weather one-hot, terrain one-hot, trickroom, hazards, stats, etc.
        obs = list()
        for p in ownActive:
            obs.append(p.hp / 255.0)
            obs.append(p.battling_moves.index(p.last_used_move) if p.last_used_move else -1)
        if len(ownActive) == 1:
            obs.append(0.0)
            obs.append(0.0)  # If only one Pokémon is active, the second is considered fainted
        obs.append(self.state.weather.value)
        obs.append(self.state.field.value) 
        obs.append(int(self.state.trickroom))  # Trickroom is a boolean, convert to int (0 or 1)
                   
        return np.array(obs, dtype=np.float32)  # Example features, adjust as needed

    def _decode_action(self, action, state_view):
        # TODO: use state view (or self.team_view?) to catch cases where one of the Pokémon is fainted etc. (basically when the action space is reduced)
        commands = []
        for a in action:  # for each active Pokémon
            if a < 8:
                move_idx = (a // 2) + 1  # moves are indexed from 1 because 0 is reserved for switch
                target_idx = a % 2
                commands.append((move_idx, target_idx))
            else:
                reserve_idx = a - 8
                commands.append((0, reserve_idx))  # switch command
        return commands

    def _get_reward(self, initialTotalHp: int):
        # Might actually not work for cases where opponent switches out Pokémon or when Pokémon faint
        reward = 0
        newHp = 0
        for p in self.state.sides[1].team.active:
            newHp += p.hp  # Assuming this method exists to get HP of Pokémon
        dmg = initialTotalHp - newHp
        if(self.engine.winning_side == 0):
            reward += 1000
        elif(self.engine.winning_side == 1):
            reward -= 1000
        return reward + dmg
    
    @staticmethod
    def encode_pkm(p: BattlingPokemon):
        feats = []
        feats.append(p.hp / 255)
        # Boosts normalized:
        feats += [b / 8 for b in p.boosts[1:8]]
        # Status one-hot or integer:
        s = p.status.value  # better: one-hot arr
        feats.append(s)  # or expand to one-hot vectors
        # Sleep turns rem:
        feats.append(p._wake_turns)
        # Protect/consec:
        feats.append(1.0 if p.protect else 0.0)
        feats.append(p._consecutive_protect)
        # Moves (only known moves – length=4):
        for m in p.battling_moves:
            feats.append(m.pp / m.constants.max_pp)
            feats.append(1.0 if m.disabled else 0.0)
        return feats

    def render(self, mode='human'):
        print(self.engine)