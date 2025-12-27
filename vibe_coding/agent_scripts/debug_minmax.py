
import sys
import os
import random
import numpy as np

# Add project src folder to PYTHONPATH
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from azul.env import AzulEnv
from players.heuristic_min_max_mcts_player import HeuristicMinMaxMCTSPlayer

def debug_minmax():
    print("=== Debugging MinMax Player ===")
    
    # Initialize Environment
    env = AzulEnv(num_players=2)
    obs = env.reset(initial=True)
    
    # Initialize Player
    # Depth 2 to keep it fast, but deep enough to see recursive logic
    player = HeuristicMinMaxMCTSPlayer(strategy='minmax', depth=2)
    
    print(f"Current Player: {env.current_player}")
    
    # Force a state where we can evaluate clearly
    # Let's say we have some factories
    print("\n--- Initial State ---")
    
    valid_actions = env.get_valid_actions()
    print(f"Number of valid actions: {len(valid_actions)}")
    
    # Trace the _minmax_search
    print("\n--- Running MinMax Search ---")
    best_action_idx = player.predict(obs)

    best_action = env.index_to_action(best_action_idx)
    print(f"Selected Action: {best_action}")
    
    # Let's verify what the evaluation function returns for this action vs others
    print("\n--- Verifying Values ---")
    
    # We will manually invoke the private _evaluate_state to see what it thinks
    # Only useful if we step into the future.
    
    # Let's check the score for the selected action
    clone = env.clone()
    clone.step(best_action)
    val = player._evaluate_state(clone) # This uses existing logic
    print(f"Value of selected state (using internal eval): {val}")
    
    # Now let's try a random other action
    other_action = valid_actions[0] if valid_actions[0] != best_action else valid_actions[1]
    print(f"Comparing with Other Action: {other_action}")
    
    clone_other = env.clone()
    clone_other.step(other_action)
    val_other = player._evaluate_state(clone_other)
    print(f"Value of other state: {val_other}")
    
    # CRITICAL TEST: Switch to Player 1 and see if they maximize THEIR score
    print("\n--- Testing Player 1 Perspective ---")
    
    # Reset and advance to P1
    env.reset()
    # P0 does something random
    valid_0 = env.get_valid_actions()
    env.step(valid_0[0])
    
    print(f"Current Player: {env.current_player} (Should be 1)")
    
    best_action_p1_idx = player.predict(env._get_obs())
    best_action_p1 = env.index_to_action(best_action_p1_idx)
    print(f"P1 Selected Action: {best_action_p1}")
    
    # Check value
    clone_p1 = env.clone()
    clone_p1.step(best_action_p1)
    
    # The internal eval logic:
    # returns (P0 + bonus0) - (P1 + bonus1)
    # If P1 is optimizing correctly, they should MINIMIZE this value (make it negative).
    
    val_p1 = player._evaluate_state(clone_p1)
    print(f"Value for P1's choice (internal, should be negative?): {val_p1}")
    
    # Compare with another random action for P1
    valid_1 = env.get_valid_actions()
    other_p1 = valid_1[0] if valid_1[0] != best_action_p1 else valid_1[1]
    
    clone_other_p1 = env.clone()
    clone_other_p1.step(other_p1)
    val_other_p1 = player._evaluate_state(clone_other_p1)
    print(f"Value for P1's OTHER choice: {val_other_p1}")
    
    if val_p1 > val_other_p1:
        print("POSSIBLE BUG: P1 chose a state with HIGHER (P0-P1) value. They should minimize P0-P1.")
    else:
        print("Behavior seems correct (P1 minimized P0-P1).")

if __name__ == "__main__":
    debug_minmax()
