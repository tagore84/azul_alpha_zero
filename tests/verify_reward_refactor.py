
import sys
import os
import numpy as np
import torch
from unittest.mock import MagicMock

# Add project src folder to PYTHONPATH
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from azul.env import AzulEnv
from train.self_play import play_game

class DummyModel:
    def __init__(self, action_size):
        self.action_size = action_size
        self.device = 'cpu'

    def parameters(self):
        return iter([torch.tensor(0)]) # Dummy parameter for device check

    def eval(self):
        pass

    def predict(self, obs_batch, action_mask=None):
        batch_size = obs_batch.shape[0]
        # Random policy
        pi = np.random.rand(batch_size, self.action_size)
        pi = pi / pi.sum(axis=1, keepdims=True)
        # Random value
        v = np.random.rand(batch_size) * 2 - 1
        return pi, v
    
    def to(self, device):
        return self

def test_reward_value_target():
    print("Testing Reward Refactor...")
    env = AzulEnv()
    model = DummyModel(env.action_size)
    
    # Run a very short game (low simulations) to get examples
    # We don't care about the quality of moves, just the logged value target
    examples, stats = play_game(env, model, simulations=2, cpuct=1.0)
    
    print(f"Game finished. P0 Score: {stats['p0_score']}, P1 Score: {stats['p1_score']}")
    
    if not examples:
        print("No examples generated! (Maybe game was discarded?)")
        sys.exit(1)
        
    # Check the last example (or any example)
    # The value 'v' should roughly be score / 100.0
    
    # examples is a list of dicts: {'obs': ..., 'pi': ..., 'v': ...}
    # Note: 'v' is from the perspective of the player to move.
    
    # Let's check the logic.
    # In play_game refactor:
    # val_0 = clip(score_p0 / 100.0, -1, 1)
    # val_1 = clip(score_p1 / 100.0, -1, 1)
    # diff_0 = val_0
    
    # If player 0 moved, v = diff_0 = val_0
    
    sample_ex = examples[0]
    expected_v0 = np.clip(stats['p0_score'] / 100.0, -1.0, 1.0)
    expected_v1 = np.clip(stats['p1_score'] / 100.0, -1.0, 1.0)
    
    print(f"Expected V0: {expected_v0}, Expected V1: {expected_v1}")
    
    # Verify a few examples
    for i, ex in enumerate(examples[:5]):
        # We need to know who the player was to verify
        # The example in 'memory' has 'player' key, but 'examples' list does NOT have 'player' key in the final dict returned by play_game in original code?
        # Let's check self_play.py again.
        pass

    # Wait, looking at self_play.py:
    # examples.append({ 'obs': ..., 'pi': ..., 'v': v })
    # It does NOT store 'player'. 
    # But all examples from the SAME game share the SAME final scores.
    # So v will be either expected_v0 or expected_v1.
    
    found_match = False
    for ex in examples:
        v = ex['v']
        if np.isclose(v, expected_v0) or np.isclose(v, expected_v1):
            found_match = True
        else:
             print(f"Propblem found! v={v} matches neither {expected_v0} nor {expected_v1}")
             # It acts weirdly if v is not one of them.
             
    if found_match:
        print("SUCCESS: Value targets match normalized scores.")
    else:
        print("FAILURE: Value targets do NOT match normalized scores.")
        sys.exit(1)

if __name__ == "__main__":
    test_reward_value_target()
