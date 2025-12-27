
import sys
import os
import numpy as np
import torch
from collections import defaultdict

# Add project src folder to PYTHONPATH
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

from train.self_play import play_game
from azul.env import AzulEnv
from net.azul_net import AzulNet
from mcts.mcts import MCTS

class MockModel:
    def __init__(self, action_size):
        self.device = torch.device('cpu')
        self.action_size = action_size
    def predict(self, obs, mask):
        # random logits and value
        batch_size = obs.shape[0]
        pi = np.random.randn(batch_size, self.action_size)
        v = np.random.randn(batch_size, 1)
        return pi, v
    def parameters(self):
        return [torch.tensor(0.0)]
    def to(self, device):
        return self
    def eval(self):
        pass

def inspect_self_play():
    env = AzulEnv()
    model = MockModel(env.action_size)
    
    print("Running one game of self-play to inspect data generation...")
    examples, stats = play_game(env, model, simulations=10, cpuct=1.0)
    
    print(f"\nGenerated {len(examples)} examples.")
    print(f"Game Winner: {stats.get('winner')}, P0 Score: {stats.get('p0_score')}, P1 Score: {stats.get('p1_score')}")
    
    for i, ex in enumerate(examples):
        print(f"\n--- Example {i} ---")
        # Decode obs
        # We can't easily decode flat obs back to dict without access to env internal structure, 
        # but we can look at the 'v' and 'player' (if we stored it, oh wait, play_game stores 'player' in memory but examples only have 'obs', 'pi', 'v')
        
        # Wait, 'play_game' logic:
        # v = diff_0 if entry['player'] == 0 else diff_1
        
        v_target = ex['v']
        print(f"Value Target (v): {v_target}")
        
        # Check if v matches the stats
        # If winner is P0 (index 0), then for player 0 v should be positive (if zero sum)
        
        # Let's try to infer whose turn it was from obs?
        # Dataset slicing: Global part has 'current_player' at end? 
        # No, 'encode_observation' puts 'current_player' implicitly by rotation?
        # Actually 'encode_observation' ROTATES players so current_player is always P0 relative to the observation.
        # So 'players[0]' in the spatial/global part is ALWAYS the current player.
        
        pass

if __name__ == "__main__":
    inspect_self_play()
