
import sys
import os
import numpy as np

sys.path.insert(0, os.path.abspath('src'))
from azul.env import AzulEnv

def test_mechanics():
    print("Initializing AzulEnv...")
    env = AzulEnv(num_players=2)
    obs = env.reset(initial=True)
    
    # Debug: Print initial factories
    print("Factories:\n", env.factories)
    
    # Try to take tiles and place in Row 0 (Size 1) -> Immediate Wall Placement -> Points
    # Find a factory with tiles
    actions = env.get_valid_actions()
    
    target_action = None
    # We want: Source < 5, Dest = 0 (Pattern Line 1)
    for a in actions:
        source, color, dest = a
        if source < 5 and dest == 0:
            target_action = a
            break
            
    if target_action is None:
        print("FATAL: No valid action to place in pattern line 0?")
        return

    print(f"Executing Action: {target_action} (Source={target_action[0]}, Color={target_action[1]}, Dest={target_action[2]})")
    
    obs, reward, done, info = env.step(target_action)
    
    p0 = env.players[0]
    print(f"P0 Score: {p0['score']}")
    print(f"P0 Wall:\n{p0['wall']}")
    print(f"P0 Pattern Lines: {p0['pattern_lines']}")
    
    if p0['score'] > 0:
        print("SUCCESS: Score increased!")
    else:
        print("FAILURE: Score did not increase. Why?")
        # Check if line 0 is full
        line0 = p0['pattern_lines'][0]
        print(f"Line 0 content: {line0}")
        # If line 0 is full (size 1), it should score at end of round?
        # WAIT. Azul Env (v5 log says "Speculative Bonuses").
        # In `step()`, we see logic: 
        # `is_full and not was_full: ... speculative_points += ...`
        # But `step` does NOT move tiles to wall immediately?
        # "Score placement and penalties" happens in `_end_round`.
        # UNLESS `step` adds "Speculative Points".
        
        # Let's check if the tile is in pattern line.
        
    # Check if we can complete a row to end game?
    # This requires multiple rounds.
    
if __name__ == "__main__":
    test_mechanics()
