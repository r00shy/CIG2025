from typing import Optional
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from vgc2.agent.battle import TreeSearchBattlePolicy
from vgc2.battle_engine import State
from vgc2.agent import BattlePolicy
from vgc2.battle_engine import BattleEngine
from vgc2.battle_engine.constants import BattleRuleParam
from vgc2.battle_engine.damage_calculator import calculate_damage
from vgc2.battle_engine.move import BattlingMove
from vgc2.battle_engine.pokemon import BattlingPokemon
from vgc2.battle_engine.team import Team
from vgc2.battle_engine.view import TeamView, StateView
from vgc2.util.generator import gen_team, _RNG
from vgc2.battle_engine.game_state import get_battle_teams
from vgc2.competition.match import label_teams  # assuming this is a util wrapper

from vgc2.agent.battle import GreedyBattlePolicy


class PokemonBattleEnv(gym.Env):
    def __init__(self, team_size=4, n_moves=5, n_active=2):
        super().__init__()
        self.team_size = team_size
        self.n_moves = n_moves
        self.n_active = n_active

        self.opponent_policy = GreedyBattlePolicy()
     
        self.action_space = spaces.MultiDiscrete([10, 10])  # why 10 moves: 4 moves * 2 possible targets + 2 switch 

        self.observation_space = gym.spaces.Box(-1, 300, shape=(304,), dtype=np.float32)
        self.total_turns = 0
        self.last_action = (0, 0)  
        self.reset()


    def reset(self, seed: Optional[int] = None):
        self.params = BattleRuleParam()
        self._rewarded_faints = set()
        self.consecutive_switches = 0
        self.last_move_used = None
        self.last_move_effectiveness = 1.0
        self.team = gen_team(self.team_size, self.n_moves), gen_team(self.team_size, self.n_moves)
        label_teams(self.team)
        self.team_view = TeamView(self.team[0]), TeamView(self.team[1])
        self.state = State(get_battle_teams(self.team, self.n_active))
        self.state_view = StateView(self.state, 0, self.team_view), StateView(self.state, 1, self.team_view)
        self.engine = BattleEngine(self.state)
        self.oppMaxHp = sum(p.hp for p in self.state.sides[1].team.active + self.state.sides[1].team.reserve)
        self.ownMaxHp = sum(p.hp for p in self.state.sides[0].team.active + self.state.sides[0].team.reserve)
        info = {}

        return (self._get_obs(), info)

    def step(self, action):
        self.total_turns += 1
        
        # Track previous state for reward calculation
        prev_own_hp = sum(p.hp for p in self.state.sides[0].team.active + self.state.sides[0].team.reserve)
        prev_opp_hp = sum(p.hp for p in self.state.sides[1].team.active + self.state.sides[1].team.reserve)
        
        # Execute actions
        agent_command = self._decode_action(action)
        opponent_command = self.opponent_policy.decision(self.state_view[1], self.team_view[0])
        self.engine.run_turn((agent_command, opponent_command, ))
        
        # Calculate new state
        new_own_hp = sum(p.hp for p in self.state.sides[0].team.active + self.state.sides[0].team.reserve)
        new_opp_hp = sum(p.hp for p in self.state.sides[1].team.active + self.state.sides[1].team.reserve)
        
        # Get observation
        obs = self._get_obs()
        terminated = self.engine.finished()
        info = {}
        
        # Calculate reward
        reward = self._get_reward(
            prev_own_hp, 
            prev_opp_hp,
            new_own_hp,
            new_opp_hp,
            action
        )
        
        # Track switching behavior
        self._update_switch_tracking(action)
        
        """ print(f"Turn: {self.engine.turn}, Reward: {reward:.2f}, Action: {agent_command}") """
        if self.engine.winning_side == 0:
            """ print("We win!") """
        elif self.engine.winning_side == 1:
            """ print("We lose!") """

        return obs, reward, terminated, False, info
    
    def _get_obs(self):
        def get_pkm_features(p):
            """Simplified Pokémon features (9 features)"""
            if not p:  # Pad for missing Pokémon
                return [0.0] * 9
                
            features = [
                p.hp / p.constants.stats[0],  # Normalized HP
                p.status.value,
                p.types[0].value if len(p.types) > 0 else 0,
                p.types[1].value if len(p.types) > 1 else 0,
                *p.boosts  # 6 stat boosts (accuracy/evasion included)
            ]
            # Pad boosts if needed (should always be 6)
            return features[:9] if len(features) > 9 else features + [0]*(9-len(features))

        def get_move_features(move: BattlingMove):
            """Essential move properties (4 features)"""
            if not move:
                return [0.0] * 4
                
            return [
                move.constants.pkm_type.value,
                move.pp / max(1, move.constants.max_pp),  # Normalized PP
                move.constants.base_power / 200.0,  # Normalized power
                move.constants.accuracy / 100.0 if move.constants.accuracy > 0 else 1.0
            ]

        obs = []
        
        # --- Core Battle State ---
        # Active Pokémon (both sides)
        for side in [0, 1]:
            for i in range(self.n_active):
                pkm = self.state.sides[side].team.active[i] if i < len(self.state.sides[side].team.active) else None
                obs.extend(get_pkm_features(pkm))
        
        # Reserve Pokémon (both sides)
        for side in [0, 1]:
            for i in range(self.team_size - self.n_active):
                pkm = self.state.sides[side].team.reserve[i] if i < len(self.state.sides[side].team.reserve) else None
                obs.extend(get_pkm_features(pkm))
        
        # Global state
        obs.append(self.state.weather.value)
        obs.append(self.state.field.value)
        obs.append(int(self.state.trickroom))
        
        # --- Move Information ---
        # Only for active Pokémon (both sides)
        for side in [0, 1]:
            for i in range(self.n_active):
                pkm = self.state.sides[side].team.active[i] if i < len(self.state.sides[side].team.active) else None
                
                # Get up to 4 moves
                moves = pkm.battling_moves[:4] if pkm else []
                for j in range(4):
                    move = moves[j] if j < len(moves) else None
                    obs.extend(get_move_features(move))
        
        # --- Type Effectiveness ---
        # Only calculate for agent's moves vs opponent's active
        for i in range(min(self.n_active, len(self.state.sides[0].team.active))):
            pkm = self.state.sides[0].team.active[i]
            moves = pkm.battling_moves[:4]
            
            for move in moves:
                if not move:
                    obs.extend([0.0] * self.n_active)
                    continue
                    
                effectiveness = []
                move_type = move.constants.pkm_type.value
                
                # Against each opponent active
                for j in range(min(self.n_active, len(self.state.sides[1].team.active))):
                    opp = self.state.sides[1].team.active[j]
                    eff = 1.0
                    for t in opp.types:
                        eff *= self.params.DAMAGE_MULTIPLICATION_ARRAY[move_type][t.value]
                    effectiveness.append(eff)
                
                # Pad if needed
                effectiveness.extend([0.0] * (self.n_active - len(effectiveness)))
                obs.extend(effectiveness)
        
        # Verify size
        if len(obs) != 304:
            # Auto-pad for safety
            obs.extend([0.0] * (304 - len(obs)))
        
        return np.array(obs, dtype=np.float32)

    def _decode_action(self, action):
        commands = []
        for a in action:  # for each active Pokémon
            if 0 <= a <= 7:
                # 0–7 are move commands
                move_idx = a // 2
                target_idx = a % 2
                commands.append((move_idx, target_idx))
            elif 8 <= a <= 9:
                # 8–9 are switch commands to reserve index 0 or 1
                reserve_idx = a - 8
                commands.append((-1, reserve_idx))
            else:
                raise ValueError(f"Invalid action value: {a}")
        return commands

    """ def _get_reward(self, action):
        reward = 0.0
        state = self.state
        
        # 1. Penalize switching actions
        switch_penalty = 0
        for a in action:  # Assuming 'action' is stored in step()
            if a >= 8:  # Switch command (8 or 9)
                switch_penalty -= 1.5  # Significant penalty per switch
        reward += switch_penalty
        
        # 2. Reward attacking behavior
        attack_bonus = 0
        for pkm in state.sides[0].team.active:
            if pkm.last_used_move and pkm.last_used_move.id != 0:  # 0 = "No Move"
                attack_bonus += 0.3
        reward += attack_bonus
        
        # 3. Relative HP advantage (primary driver)
        current_own_hp = sum(p.hp for p in state.sides[0].team.active + state.sides[0].team.reserve)
        current_opp_hp = sum(p.hp for p in state.sides[1].team.active + state.sides[1].team.reserve)
        hp_advantage = (current_own_hp / self.ownMaxHp) - (current_opp_hp / self.oppMaxHp)
        reward += hp_advantage * 1.0  # Reduced weight to balance with other factors
        
        # 4. Type effectiveness bonus
        type_bonus = 0
        for pkm in state.sides[0].team.active:
            if pkm.last_used_move:
                move_type = pkm.last_used_move.constants.pkm_type.value
                for opp in state.sides[1].team.active:
                    effectiveness = 1.0
                    for t in opp.types:
                        effectiveness *= self.params.DAMAGE_MULTIPLICATION_ARRAY[move_type][t.value]
                    # Scale bonus based on effectiveness
                    if effectiveness > 1.0:
                        type_bonus += 0.5 * (effectiveness - 1.0)
                    elif effectiveness < 1.0:
                        type_bonus -= 0.2 * (1.0 - effectiveness)
        reward += type_bonus
        
        # 5. Faint rewards (scaled by importance)
        for pkm in state.sides[1].team.active + state.sides[1].team.reserve:
            if pkm.fainted and pkm not in self._rewarded_faints:
                max_hp = pkm.constants.stats[0]
                reward += (max_hp / self.oppMaxHp) * 3.0
                self._rewarded_faints.add(pkm)
        
        for pkm in state.sides[0].team.active + state.sides[0].team.reserve:
            if pkm.fainted and pkm not in self._rewarded_faints:
                max_hp = pkm.constants.stats[0]
                reward -= (max_hp / self.ownMaxHp) * 3.0
                self._rewarded_faints.add(pkm)
        
        # 6. Consecutive switch penalty
        if hasattr(self, 'consecutive_switches'):
            if any(a >= 8 for a in action):
                self.consecutive_switches += 1
            else:
                self.consecutive_switches = 0
        else:
            self.consecutive_switches = 0
        
        if self.consecutive_switches > 1:
            reward -= 0.5 * self.consecutive_switches
        
        # 7. Progress and terminal rewards
        if self.engine.winning_side == 0:
            reward += 20.0
        elif self.engine.winning_side == 1:
            reward -= 20.0
        
        # 8. Time penalty (encourage decisive actions)
        reward -= 0.01
        
        return reward """
    def _get_reward(self, prev_own_hp, prev_opp_hp, new_own_hp, new_opp_hp, action):
        """Simplified and focused reward function"""
        reward = 0.0
        
        # 1. Core battle performance (60% weight)
        damage_dealt = (prev_opp_hp - new_opp_hp) / self.oppMaxHp
        damage_taken = (prev_own_hp - new_own_hp) / self.ownMaxHp
        reward += 3.0 * damage_dealt
        reward -= 2.0 * damage_taken
        
        # 2. Battle outcome rewards (30% weight)
        if self.engine.winning_side == 0:
            reward += 15.0
        elif self.engine.winning_side == 1:
            reward -= 15.0
            
        # 3. Action quality incentives (10% weight)
        # Penalize consecutive switches
        if any(a >= 8 for a in action):
            reward -= 1.5 * (self.consecutive_switches + 1)
            
        # Reward effective move usage
        if self.last_move_used and self.last_move_effectiveness > 1.0:
            reward += 0.8 * (self.last_move_effectiveness - 1.0)
            
        # Small penalty per turn to encourage decisive actions
        reward -= 0.05
        
        return reward

    def _update_switch_tracking(self, action):
        """Track switching patterns to prevent loops"""
        if any(a >= 8 for a in action):  # Switch action
            self.consecutive_switches += 1
        else:
            self.consecutive_switches = 0
            
        # Track last move effectiveness
        self.last_move_used = None
        self.last_move_effectiveness = 1.0
        for pkm in self.state.sides[0].team.active:
            if pkm.last_used_move:
                self.last_move_used = pkm.last_used_move
                # Calculate effectiveness against current opponents
                for opp in self.state.sides[1].team.active:
                    effectiveness = 1.0
                    for t in opp.types:
                        effectiveness *= self.params.DAMAGE_MULTIPLICATION_ARRAY[
                            self.last_move_used.constants.pkm_type.value
                        ][t.value]
                    self.last_move_effectiveness = max(
                        self.last_move_effectiveness, 
                        effectiveness
                    )
    
    def get_action_mask(self):
        masks = []
        for i in range(2):  # For each active Pokémon
            pokemon_mask = []
            if i < len(self.state.sides[0].team.active):
                pkm = self.state.sides[0].team.active[i]
                
                # Move actions
                for move_idx in range(4):
                    if move_idx < len(pkm.battling_moves):
                        move = pkm.battling_moves[move_idx]
                        # Valid if PP > 0 and not disabled
                        valid = move.pp > 0 and not move.disabled
                        # Add for both targets
                        pokemon_mask.append(valid)
                        pokemon_mask.append(valid)
                    else:
                        pokemon_mask.extend([False, False])  # Invalid move slot
                
                # Switch actions
                valid_switches = [False, False]
                for switch_idx in range(2):
                    if (switch_idx < len(self.state.sides[0].team.reserve) and not self.state.sides[0].team.reserve[switch_idx].fainted):
                        valid_switches[switch_idx] = True
                pokemon_mask.extend(valid_switches)
            else:
                pokemon_mask = [False] * 10  # No Pokémon in slot
                
            masks.append(pokemon_mask)
        return masks
        