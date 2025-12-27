
import numpy as np
from azul.env import AzulEnv

def test_first_player_token_bug():
    env = AzulEnv(num_players=2)
    env.reset()
    
    # Force a state where center has tiles and it's the first time taking from center
    # We need to simulate a few moves to put tiles in center
    
    # Manually set factories to have some tiles
    env.factories[0] = np.array([1, 1, 0, 0, 0]) # 1 Blue, 1 Yellow
    
    # Player 0 takes Blue (0) from Factory 0
    # This should put Yellow (1) into Center
    print("Step 1: Player 0 takes Blue from Factory 0")
    obs, reward, done, info = env.step((0, 0, 0)) # Source 0, Color 0, Dest 0
    
    print(f"Center after Step 1: {env.center}")
    assert env.center[1] == 1, "Center should have 1 Yellow tile"
    assert env.first_player_token == True, "First player token should be present"
    
    # Player 1 takes Yellow (1) from Center (Source N)
    # This should trigger the penalty
    print("Step 2: Player 1 takes Yellow from Center")
    
    # Check Player 1's floor line before
    p1_floor_before = env.players[1]['floor_line'].copy()
    print(f"Player 1 Floor Before: {p1_floor_before}")
    
    obs, reward, done, info = env.step((env.N, 1, 1)) # Source Center, Color 1, Dest 1
    
    p1_floor_after = env.players[1]['floor_line'].copy()
    print(f"Player 1 Floor After: {p1_floor_after}")
    
    # The bug is that the token is not added, or added incorrectly
    # In the buggy version, it tries to find '0' (Blue) instead of '-1' (Empty) to replace with '-1' (Empty)
    # So if the floor was empty (-1, -1, ...), nothing happens or it stays -1.
    
    # We expect to see a penalty token. Since we haven't defined a value yet, 
    # let's see what happens. If the bug is present, the floor line might be unchanged 
    # (except for the tiles we just took, if any overflowed, but here we took 1 tile to line 1, capacity 2, so no overflow from tiles)
    
    # Wait, if we took from center, we get the penalty.
    # The penalty should be in the floor line.
    
    # In the buggy code:
    # idxs = np.where(fl == 0)[0]  <-- Looks for 0 (Blue)
    # if idxs.size > 0: fl[idxs[0]] = -1 <-- Sets to -1
    
    # Since floor is initialized to -1, and we haven't put any Blue (0) tiles there,
    # np.where(fl == 0) will be empty.
    # So nothing happens.
    
    # So P1 floor should be all -1 (if no overflow).
    # But we expect the penalty token!
    
    has_penalty = False
    # We don't know what the penalty value IS yet, but it should NOT be all -1.
    if np.any(p1_floor_after != -1):
        has_penalty = True
        print("Penalty token found (or some tile on floor).")
    else:
        print("BUG CONFIRMED: No penalty token found on floor.")
        
    assert has_penalty, "FAIL: First player token penalty was not applied!"
    print("SUCCESS: Penalty token applied (Bug fixed).")

if __name__ == "__main__":
    try:
        test_first_player_token_bug()
    except AssertionError as e:
        print(e)
        exit(1)
    except Exception as e:
        print(f"An error occurred: {e}")
        exit(1)
