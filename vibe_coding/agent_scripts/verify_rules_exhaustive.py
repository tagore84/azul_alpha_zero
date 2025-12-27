import numpy as np
import sys
import os

# Add src to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from azul.env import AzulEnv

def test_first_player_token_floor_full():
    print("\n--- Test: First Player Token vs Full Floor ---")
    env = AzulEnv(num_players=2, seed=42)
    env.reset()
    
    # 1. Fill Player 0's floor line completely
    # Floor line size is 7
    p0 = env.players[0]
    p0['floor_line'][:] = 1 # Fill with color 1 (Yellow)
    print(f"Initial Floor Line: {p0['floor_line']}")
    
    # 2. Put some tiles in Center (simulate a factory dump)
    env.center[0] = 1 # 1 Blue tile in center
    env.first_player_token = True
    
    # 3. Action: Player 0 takes Blue (0) from Center (5)
    # Action tuple: (source, color, dest) -> (5, 0, 0)
    # Dest 0 is pattern line 0.
    action = (5, 0, 0)
    
    print(f"Player 0 takes from center. First token present: {env.first_player_token}")
    env.current_player = 0
    obs, reward, done, info = env.step(action)
    
    p0 = env.players[0]
    print(f"Post-Action Floor Line: {p0['floor_line']}")
    
    # Check if First Player Token (Value 5) is in floor line OR if penalty was applied
    has_token = (p0['floor_line'] == 5).any()
    
    if has_token:
        print("SUCCESS: Token 5 found in floor line (replaced something or added).")
    else:
        # If not in floor line, did they get the penalty for it?
        # A full floor of 7 tiles gives: -1-1-2-2-2-3-3 = -14 points.
        # With FP token, it should be... actually standard rules say 'overflow tiles go to lid'.
        # But the FP token specifically: "If you have the starting player marker in your floor line... Note: If you have the starting player marker in your floor line, it counts as a normal tile there."
        # If floor is full, where does it go?
        # Rulebook: "If all spaces of your floor line are occupied, return any further fallen tiles to the lid... Note: If you have the starting player marker in your floor line, it counts as a normal tile there. But instead of placing it in the lid, place it in front of you."
        # This implies it DOES count as penalty even if floor is full!
        print("FAILURE? Token 5 NOT found in floor line.")
        print("Checking if expected penalty logic handles this (maybe it's tracked separately?)")
        if env.first_player_token == False:
             print("Token was taken (env.first_player_token is False).")
        else:
             print("Token was NOT taken.")

def test_infinite_refill():
    print("\n--- Test: Infinite Refill (Empty Bag/Discard) ---")
    env = AzulEnv(num_players=2, seed=42)
    env.reset()
    
    # Empty Bag and Discard
    env.bag[:] = 0
    env.discard[:] = 0
    
    print(f"Bag: {env.bag}, Discard: {env.discard}")
    
    # Trigger Refill
    print("Triggering refill...")
    env._refill_factories()
    
    factories_sum = env.factories.sum()
    print(f"Factories Sum after refill: {factories_sum}")
    
    if factories_sum > 0:
        print("FAILURE: Factories refilled despite empty bag/discard! (Infinite tiles bug)")
    else:
        print("SUCCESS: Factories stayed empty.")

if __name__ == "__main__":
    test_first_player_token_floor_full()
    test_infinite_refill()
