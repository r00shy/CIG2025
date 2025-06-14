import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from vgc2.agent import BattlePolicy
from vgc2.battle_engine import State, BattleCommand, TeamView
from typing import Optional

# Simple feedforward neural network for battle decision
class BattleNet(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

# Feature extraction from battle state (improved: uses opp_view if available)
def extract_features(state: State, opp_view: Optional[TeamView] = None) -> np.ndarray:
    features = []
    # Own side features
    for pkm in state.sides[0].team.active:
        features.append(pkm.hp / pkm.constants.stats[0])  # normalized HP
        features.extend(list(pkm.constants.stats))        # base stats
    # Opponent side features
    if opp_view is not None:
        for pkm in opp_view.members:
            features.append(pkm.hp / pkm.stats[0] if hasattr(pkm, 'hp') and hasattr(pkm, 'stats') else 0)
            features.extend(list(pkm.stats) if hasattr(pkm, 'stats') else [0]*6)
    else:
        for pkm in state.sides[1].team.active:
            features.append(pkm.hp / pkm.constants.stats[0])
            features.extend(list(pkm.constants.stats))
    return np.array(features, dtype=np.float32)

class DoubleLNeuralBattlePolicy(BattlePolicy):
    """
    Battle policy using a neural network to select the best action.
    """
    def __init__(self, input_dim=14, output_dim=8, model_path=None):
        self.model = BattleNet(input_dim, output_dim)
        if model_path:
            self.model.load_state_dict(torch.load(model_path))
        self.model.eval()

    def decision(self, state: State, opp_view: Optional[TeamView] = None) -> list[BattleCommand]:
        # Extract features (now uses opp_view)
        features = extract_features(state, opp_view)
        x = torch.tensor(features).unsqueeze(0)  # batch dimension
        with torch.no_grad():
            logits = self.model(x)
            action_idx = torch.argmax(logits, dim=1).item()
        # Map action_idx to a valid BattleCommand (example: pick move 0, target 0, etc.)
        # This mapping should be improved for your use case
        # Here, we assume single battle, 4 moves, 2 possible targets
        move = action_idx % 4
        target = (action_idx // 4) % 2
        return [(move, target)]

# Example usage:
# policy = DoubleLNeuralBattlePolicy()
# cmds = policy.decision(state)
