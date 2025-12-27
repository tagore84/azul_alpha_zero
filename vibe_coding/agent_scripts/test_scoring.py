import sys
import os
import numpy as np

# Add src to python path
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))
from azul.env import AzulEnv

def test_floor_penalty_immediate():
    print("--- Test Floor Penalty Immediate ---")
    env = AzulEnv()
    env.reset()
    
    # Player 0 takes from factory to floor (or overflow)
    # Let's force some tiles to floor.
    # We can hack the state or use actions.
    # Using actions is better.
    
    # Factory 0 has 4 tiles. Let's make Player 0 take them to Floor (Dest 5).
    # source=0, color=?, dest=5
    f0 = env.factories[0]
    color = np.argmax(f0 > 0)
    count = f0[color]
    print(f"Player 0 moving {count} tiles of color {color} to Floor")
    
    # Initial score
    print(f"Initial Score: {env.players[0]['score']}")
    
    action = (0, int(color), 5)
    env.step(action)
    
    new_score = env.players[0]['score']
    print(f"New Score: {new_score}")
    
    # Expected penalty for N tiles: -1, -1, -2, -2...
    # If 4 tiles: -1 -1 -2 -2 = -6? 
    # Let's check rules.py penalties list: [-1, -1, -2, -2, -2, -3, -3]
    expected_pen = 0
    penalties = [-1, -1, -2, -2, -2, -3, -3]
    for i in range(count):
        expected_pen += penalties[i]
        
    print(f"Expected Penalty: {expected_pen}")
    
    if new_score == expected_pen:
        print("PASS: Score updated immediately correctly.")
    else:
        print(f"FAIL: Score mismatch. Got {new_score}, expected {expected_pen}")

    # Now verify end of round doesn't double count
    # We need to finish the round. 
    # Just force end round logic by clearing factories
    env.factories[:] = 0
    env.center[:] = 0
    
    print("Forcing end of round...")
    # env.step((0,0,5)) <- REMOVED invalid call
    
    # Let's call _end_round manually for testing env logic
    env._end_round()
    
    final_score = env.players[0]['score']
    print(f"Final Score after end_round: {final_score}")
    
    if final_score == expected_pen:
         print("PASS: Final score matches speculative score (no double counting).")
    else:
         print(f"FAIL: Final score mismatch. Got {final_score}, expected {expected_pen}")

def test_wall_points_speculative():
    print("\n--- Test Wall Points Speculative ---")
    env = AzulEnv()
    env.reset()
    
    # Setup: Player 0 has 4 tiles of color 0 in row 4 (needs 5)
    # AND the wall row 4 is almost full (needs only the Color 0 slot)
    # Row 4 Pattern: [Yellow, Orange, Black, Red, Blue] (Indices 0,1,2,3,4)
    # Red (Color 0) is at index 3?? No, let's check rules.py
    # rules.py: row_patterns[4] = [Yellow, Orange, Black, Red, Blue]
    # Color 0 (Blue) is at index 4.
    # Color 4 (Red) is at index 3.
    # In my test above I used Color 0.
    # Wait, Color(0) is Blue.
    # So Color 0 is at index 4 in Row 4.
    # So we need to fill indices 0,1,2,3.
    
    # Pre-fill wall indices 0,1,2,3 with dummy values (e.g. their colors)
    # We just need them to be != -1
    for c in range(4):
        env.players[0]['wall'][4][c] = 99 # Dummy color, just not -1
    
    # Pre-fill pattern line with 4 Blues (Color 0)
    # We need 5 items. 4 are there.
    env.players[0]['pattern_lines'][4][0:4] = 0 # Color 0 (Blue)
    
    print(f"Setup: Row 4 Pattern has 4/5 Blue tiles. Wall Row 4 has 4/5 filled.")
    
    # Action: Take 1 Red tile (Color 0) to Row 4 (Dest 4)
    # Put 1 Red in Factory 0
    env.factories[:] = 0
    env.center[:] = 0
    env.factories[0][0] = 1 
    
    action = (0, 0, 4)
    env.step(action)
    
    new_score = env.players[0]['score']
    print(f"New Score: {new_score}")
    
    # Expected: 
    # Placement: The new tile completes a row of 5. 
    # Horizontal segment: 1 (new) + 4 (neighbors) = 5 points.
    # Vertical segment: 0.
    # Total Placement = 5.
    # Speculative Bonus: Row Complete = +2.
    # Total = 7.
    expected_points = 7
    
    if new_score == expected_points:
        print("PASS: Score updated immediately to 7 (5 Placement + 2 Bonus).")
    else:
        print(f"FAIL: Score mismatch. Got {new_score}, expected {expected_points}")

    # Verify Final Score
    # Note: step() likely triggered _end_round() because factories are empty.
    # So Final Score should already be set (and matched speculative).
    # If we call _end_round again, we double count.
    
    if not env.done:
        env._end_round()
    
    final_score = env.players[0]['score']
    print(f"Final Score (Reconciled): {final_score}")
    
    # Game IS over (row complete). So Bonus +2 is kept.
    # Placement +5 is kept.
    # Total 7.
    if final_score == 7:
         print("PASS: Final score correct (7).")
    else:
         print(f"FAIL: Final score mismatch. Got {final_score}, expected 7")

if __name__ == "__main__":
    test_floor_penalty_immediate()
    test_wall_points_speculative()
