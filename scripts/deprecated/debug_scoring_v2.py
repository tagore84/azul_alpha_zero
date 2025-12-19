
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from azul.env import AzulEnv
from azul.rules import calculate_floor_penalization
import numpy as np

def debug_scoring():
    env = AzulEnv()
    env.reset()
    
    print("=== Debug Scoring Script ===")
    
    # Force some tiles on floor line for P0
    p0 = env.players[0]
    p0['floor_line'][0] = 0 # -1 pt
    p0['floor_line'][1] = 1 # -1 pt
    print(f"Initial Floor Line: {p0['floor_line']}")
    print(f"Initial Score: {p0['score']}")
    
    # Calculate penalty manually
    pen = calculate_floor_penalization(p0['floor_line'])
    print(f"Calculated Penalty (Manual): {pen}") # Should be -2
    
    # Simulate a step that adds to floor
    # We need to find a valid action that fills the floor
    # Or just manipulate the state and call step with a dummy action that adds to floor?
    # No, let's just trace the logic in a synthetic way or use a real action.
    
    # Let's force a state where taking from factory 0 adds to floor
    env.factories[0] = [1, 0, 0, 0, 0] # 1 tile of color 0
    # P0 takes color 0 to floor (dest 5)
    action = (0, 0, 5) 
    
    print(f"\nExtcuting Action: {action} (Source 0, Color 0, Dest Floor)")
    obs, reward, done, info = env.step(action)
    
    print(f"New Floor Line: {env.players[0]['floor_line']}")
    print(f"New Score: {env.players[0]['score']}")
    print(f"Reward: {reward}")
    
    # Check if penalty was applied correctly
    # Old pen = -2. New pen (should be -1, -1, -2) = -4. Delta = -2.
    # Score should decrease by 2.
    
    # Now let's simulate end of round
    print("\n--- Simulating End of Round ---")
    # Force round end condition
    env.factories[:] = 0
    env.center[:] = 0
    
    print(f"Score before end round: {env.players[0]['score']}")
    print(f"Accumulated score this round: {env.round_accumulated_score[0]}")
    
    # Manually trigger end round logic (simplified from env.step)
    # env.step checks _is_round_over -> _end_round
    # We can just call _end_round directly for debugging? No it's private.
    # Let's call step again with a dummy action? No, valid actions will be empty.
    # We can just inspect the logic: 
    # 1. Revert accumulated score.
    # 2. Apply final penalties.
    
    # Let's see what happens if we call _end_round
    is_done = env._end_round()
    print(f"Score after end round: {env.players[0]['score']}")
    
    # Logic trace:
    # 1. Revert: Score = Score - Accum. (If Accum is -2, Score goes back to Initial 0)
    # 2. Apply Pen: Floor line has 3 tiles (-1, -1, -2) -> -4.
    # Final Score should be 0 - 4 = -4.
    
    # If the logic in env.py says:
    # p['score'] += pen
    # And pen is -4.
    # Then final score is -4.
    
    # Wait, did we double count?
    # Speculative: Score = 0 - 2 = -2. Accum = -2.
    # End Round:
    #   Score -= Accum => -2 - (-2) = 0.
    #   Score += Pen => 0 + (-4) = -4.
    # This seems correct IF the floor line persists exactly as is.
    
    # Let's look at a scenario where tiles are CLEARED from floor.
    # Floor line is CLEARED at end of round.
    # So next round starts with empty floor.
    
    print(f"Floor line after end round: {env.players[0]['floor_line']}")
    
if __name__ == "__main__":
    debug_scoring()
