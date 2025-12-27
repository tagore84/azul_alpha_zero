
import sys
import os
import torch
import numpy as np

# Add src to python path
sys.path.append(os.path.abspath("src"))

from azul.env import AzulEnv
from net.azul_net import AzulNet
from mcts.mcts import MCTS

def load_latest_model():
    checkpoint_path = "data/checkpoints_v5/model_cycle_6.pt"
    # Basic params based on logs
    net = AzulNet(in_channels=20, global_size=38, action_size=180, factories_count=5)
    
    if os.path.exists(checkpoint_path):
        print(f"Loading checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        try:
            net.load_state_dict(checkpoint['model_state'])
        except Exception as e:
            print(f"Failed to load checkpoint (expected due to architecture change): {e}")
            print("Using untrained model for structure verification.")
    else:
        print("Checkpoint not found!")
    
    net.eval()
    return net

def print_top_actions(env, pi, top_k=5):
    valid_actions = env.get_valid_actions()
    # Create map from action tuple to prob
    action_probs = []
    
    # pi is for all actions, we need to map back
    for i, p in enumerate(pi):
        if p > 0.001:
            action = env.index_to_action(i)
            # Check validity again just to be sure
            if action in valid_actions:
                 action_probs.append((action, p))
    
    action_probs.sort(key=lambda x: x[1], reverse=True)
    
    print(f"Top {top_k} Actions:")
    for i in range(min(top_k, len(action_probs))):
        action, p = action_probs[i]
        source, color, dest = action
        source_name = f"Factory {source}" if source < 5 else "Center"
        print(f"  {i+1}. {source_name}, Color {color} -> Line {dest} (Prob: {p:.4f})")

def test_complete_row(net):
    print("\n=== Test 1: Complete Row (Win Condition) ===")
    env = AzulEnv()
    
    # Setup state: Player 0 has 4 tiles of color 0 in row 4 (needs 5)
    # Row 4 (index 4) has 5 slots.
    p = env.players[0]
    # Fill 4 slots with color 0
    p['pattern_lines'][4][0:4] = 0 
    
    # Setup Factory 0 with 1 tile of color 0 and 3 of color 1
    env.factories[0] = [1, 3, 0, 0, 0] # 1 red, 3 blue
    
    # Debug print
    # env.render()
    
    # Valid actions for Factory 0, Color 0
    # Taking 1 Red (Color 0) from Factory 0 should go to Line 4 (Index 4) to complete it.
    target_action = (0, 0, 4) 
    
    print("Running MCTS...")
    mcts = MCTS(env, net, simulations=200, cpuct=1.25)
    mcts.run()
    
    # Get Policy from MCTS
    root = mcts.root
    visits = np.zeros(env.action_size)
    for action, node in root.children.items():
        idx = env.action_to_index(action)
        visits[idx] = node.visits
    
    pi = visits / visits.sum()
    
    print_top_actions(env, pi)
    
    # Check if target action is preferred
    target_idx = env.action_to_index(target_action)
    print(f"Target Action {target_action} Prob: {pi[target_idx]:.4f}")
    
    # Get Raw Network Prior
    obs = env._get_obs()
    flat = env.encode_observation(obs)
    mask = env.get_action_mask()
    logits, value = net.predict(np.array([flat]), np.array([mask]))
    priors = np.exp(logits[0]) * mask # Naive softmax approx after masking
    priors /= priors.sum()
    
    print(f"Network Prior for Target: {priors[target_idx]:.4f}")
    print(f"Network Value Estimation: {value[0]:.4f}")

def test_avoid_floor_penalty(net):
    print("\n=== Test 2: Avoid Massive Floor Penalty ===")
    env = AzulEnv()
    
    # Setup: Player 0 has Pattern Line 0 almost full? 
    # Let's say Player 0 has a full floor line (-14 pts if takes more)
    # Wait, if floor is full, new tiles go to discard?
    # env.py: "if idxs.size > 0: ... else: self.discard[color] += 1"
    # So actually, if floor is full, it's safeish? Only penalty for things already there?
    # No, taking tiles adds to floor first.
    # Actually, once floor is full, you can't add more to it in `place_on_pattern_line` overflow.
    # But `calculate_floor_penalization` is based on what's IN the floor line.
    
    # Let's verify rule: "If the floor line is full, the excess tiles are returned to the lid/discard".
    # But the penalty is calculated based on filled slots.
    
    # Scenario: Agent has empty floor line.
    # Factory 0 has 4 tiles of Color 0.
    # Pattern Line 0 (size 1) has 0 tiles.
    # Taking 4 tiles of Color 0 -> 1 goes to Pattern Line 0, 3 overflow to Floor.
    # Floor penalty for 3 tiles: -1, -1, -2 = -4 points.
    
    # Compare with: Factory 1 has 1 tile of Color 1.
    # Taking 1 tile of Color 1 -> Goes to Pattern Line 1 (size 2). No penalty.
    
    p = env.players[0]
    env.factories[0] = [4, 0, 0, 0, 0] # 4 Reds
    env.factories[1] = [0, 1, 0, 0, 0] # 1 Blue
    
    # Target: Avoid taking Red to Line 0. Prefer Taking Blue to Line 1.
    bad_action = (0, 0, 0) # Fac 0, Red, Line 0 (Overflows 3)
    good_action = (1, 1, 1) # Fac 1, Blue, Line 1 (Safe)
    
    print("Running MCTS...")
    mcts = MCTS(env, net, simulations=200, cpuct=1.25)
    mcts.run()
    
    root = mcts.root
    visits = np.zeros(env.action_size)
    for action, node in root.children.items():
        idx = env.action_to_index(action)
        visits[idx] = node.visits
    pi = visits / visits.sum()
    
    print_top_actions(env, pi)
    
    bad_idx = env.action_to_index(bad_action)
    good_idx = env.action_to_index(good_action)
    
    print(f"Bad Action (Overflow) Prob: {pi[bad_idx]:.4f}")
    print(f"Good Action (Safe) Prob: {pi[good_idx]:.4f}")
    
    # Network Value
    obs = env._get_obs()
    flat = env.encode_observation(obs)
    mask = env.get_action_mask()
    logits, value = net.predict(np.array([flat]), np.array([mask]))
    print(f"Network Value Estimation: {value[0]:.4f}")

if __name__ == "__main__":
    net = load_latest_model()
    test_complete_row(net)
    test_avoid_floor_penalty(net)
