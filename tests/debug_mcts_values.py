
import sys
import os
import torch
import numpy as np
from typing import Tuple

# Add project src folder to PYTHONPATH
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from azul.env import AzulEnv
from net.azul_net import AzulNet
from mcts.mcts import MCTS

def decode_action(env, action_idx):
    return env.index_to_action(action_idx)

def print_tree_stats(mcts, top_k=10):
    root = mcts.root
    print(f"\n[MCTS Stats] Visits: {root.visits}, Value(Avg): {root.value:.4f}")
    
    # Sort children by visits
    children = sorted(root.children.items(), key=lambda x: x[1].visits, reverse=True)
    
    print(f"{'Action':<30} | {'Visits':<8} | {'ValSum':<10} | {'AvgVal':<10} | {'Prior':<8} | {'Q+U':<8}")
    print("-" * 90)
    
    for action, node in children[:top_k]:
        avg_val = node.value_sum / node.visits if node.visits > 0 else 0.0
        # Recompute UCB for display
        ucb = node.ucb_score(mcts.cpuct)
        
        # Format action
        # Action is tuple (factory, color, line)
        act_str = f"F{action[0]} C{action[1]} L{action[2]}"
        
        print(f"{act_str:<30} | {node.visits:<8} | {node.value_sum:<10.2f} | {avg_val:<10.4f} | {node.prior:<8.4f} | {ucb:<8.4f}")

def main():
    # 1. Load Model
    checkpoint_path = "data/checkpoints_v5/model_cycle_3.pt"
    if not os.path.exists(checkpoint_path):
        print(f"Checkpoint not found: {checkpoint_path}")
        return

    print(f"Loading model from {checkpoint_path}...")
    device = torch.device('cpu') # Debug on CPU
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Initialize Env to get shapes
    env = AzulEnv()
    obs = env.reset()
    
    # Reconstruct Model architecture
    # Assuming standard shapes from training loop
    obs_flat = env.encode_observation(obs)
    total_obs_size = obs_flat.shape[0]
    in_channels = 4
    spatial_size = 100
    factories_size = (env.N + 1) * 5
    global_size = total_obs_size - spatial_size - factories_size
    
    model = AzulNet(
        in_channels=in_channels,
        global_size=global_size,
        action_size=env.action_size,
        factories_count=env.N
    ).to(device)
    
    model.load_state_dict(checkpoint['model_state'])
    model.eval()
    
    # 2. Setup Scenario
    # Let's create a situation where taking tiles to floor is BAD, and fitting in line is GOOD.
    print("\n=== DEBUG SCENARIO ===")
    env = AzulEnv(seed=42)
    env.reset()
    
    # Let's manually manipulate state if needed, or just look at initial state.
    # Initial state: Factories full. P0 turn.
    # P0 stats: Pattern lines empty. Floor empty.
    
    # Run MCTS
    print("Running MCTS with Single Player Mode = True...")
    mcts = MCTS(env, model, simulations=50, cpuct=1.0)
    mcts.single_player_mode = True
    
    mcts.run()
    
    print_tree_stats(mcts)
    
    # Check predictions directly
    print("\n[Direct Network Prediction]")
    obs_flat = env.encode_observation(env._get_obs())
    valid = env.get_valid_actions()
    mask = np.zeros(env.action_size, dtype=np.float32)
    for a in valid:
        mask[env.action_to_index(a)] = 1.0
        
    logits, value = model.predict(np.array([obs_flat]), np.array([mask]))
    print(f"Root Value Estimate (V): {value[0]:.4f}")
    
    # Let's force a "Bad Move" (To Floor) and see evaluation
    # Find a floor move
    floor_actions = [a for a in valid if a[2] == 5]
    if floor_actions:
        bad_action = floor_actions[0]
        print(f"\nSimulating BAD move (Floor): {bad_action}")
        env_bad = env.clone()
        env_bad.step(bad_action)
        
        # Evaluate state after bad move
        obs_bad = env_bad.encode_observation(env_bad._get_obs())
        _, val_bad = model.predict(np.array([obs_bad]), np.array([mask])) # Mask irrelevant for value
        print(f"Value after BAD move (V'): {val_bad[0]:.4f}")
        # In Single Player Mode with No Flip, we expect V' < V (since state is worse)
    
    # Force a "Good Move" (To Line 0)
    good_actions = [a for a in valid if a[2] == 0]
    if good_actions:
        good_action = good_actions[0]
        print(f"\nSimulating GOOD move (Line 0): {good_action}")
        env_good = env.clone()
        env_good.step(good_action)
        
        obs_good = env_good.encode_observation(env_good._get_obs())
        _, val_good = model.predict(np.array([obs_good]), np.array([mask]))
        print(f"Value after GOOD move (V''): {val_good[0]:.4f}")
        # We expect V'' > V'

if __name__ == "__main__":
    main()
