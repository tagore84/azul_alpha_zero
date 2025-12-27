
import numpy as np
from azul.env import AzulEnv
from azul.rules import Color

def test_bug1_first_player_token_persistence():
    print("\n--- Testing Bug 1: First Player Token Persistence ---")
    env = AzulEnv(num_players=2)
    env.reset()
    
    # Setup exact state from log around line 219 (End of Round 1)
    # Player 0 has [5, 0, 0, 0, 0, -1, -1]
    env.players[0]['floor_line'] = np.array([5, 0, 0, 0, 0, -1, -1])
    # Player 1 has [-1, -1, -1, -1, -1, -1, -1] (Empty)
    env.players[1]['floor_line'] = np.full(7, -1)
    
    print(f"Before end_round P0: {env.players[0]['floor_line']}")
    print(f"Before end_round P1: {env.players[1]['floor_line']}")
    
    # Call end_round
    # We need to mock factories/center empty to ensure logic runs if we called step, 
    # but here we call _end_round directly.
    env._end_round()
    
    print(f"After end_round P0: {env.players[0]['floor_line']}")
    print(f"After end_round P1: {env.players[1]['floor_line']}")
    
    # Check if P1 has 5?
    if 5 in env.players[1]['floor_line']:
        print("FAIL: Player 1 has token 5! (Bug reproduced)")
    elif 5 in env.players[0]['floor_line']:
        print("FAIL: Player 0 still has token 5!")
    else:
        print("PASS: Floor lines cleared.")

def test_bug2_impossible_center_counts():
    print("\n--- Testing Bug 2: Impossible Center Counts ---")
    env = AzulEnv(num_players=2)
    env.reset()
    
    # Manually fill factories to a known state
    # Factory 0: 4 Blue
    env.factories[0] = [4, 0, 0, 0, 0]
    # Factory 1: 4 Blue
    env.factories[1] = [4, 0, 0, 0, 0]
    
    print(f"Initial Center: {env.center}")
    print(f"Factory 0: {env.factories[0]}")
    print(f"Factory 1: {env.factories[1]}")
    
    # Player 0 takes Blue from Factory 0
    # Should move 0 to center (none)
    env.step((0, 0, 0)) 
    print(f"After P0 takes Blue from F0 -> Center: {env.center}")
    
    # Player 1 takes Blue from Factory 1
    # Should move 0 to center
    env.step((1, 0, 1))
    print(f"After P1 takes Blue from F1 -> Center: {env.center}")
    
    # Now let's try to induce the bug. 
    # The bug report says "Center: [1 1 0 1 6]". 6 is impossible if max tiles per color is 20 and they are distributed.
    # But locally, if we have many factories with same color...
    # Let's try to put residues in center.
    
    env.reset()
    # F0: 1 Blue, 3 Red
    env.factories[0] = [1, 0, 0, 0, 3]
    # F1: 1 Blue, 3 Red
    env.factories[1] = [1, 0, 0, 0, 3]
    
    print(f"Reset. F0: {env.factories[0]}, F1: {env.factories[1]}")
    
    # P0 takes Red from F0 -> 1 Blue goes to Center
    env.step((0, 4, 0))
    print(f"After P0 takes Red from F0 -> Center: {env.center}")
    
    # P1 takes Red from F1 -> 1 Blue goes to Center. Center should have 2 Blue.
    env.step((1, 4, 1))
    print(f"After P1 takes Red from F1 -> Center: {env.center}")
    
    if env.center[0] == 2:
        print("PASS: Center accumulation seems correct so far.")
    else:
        print(f"FAIL: Center has {env.center[0]} Blue, expected 2.")

def test_bug3_illegal_actions_wall_color():
    print("\n--- Testing Bug 3: Illegal Actions (Wall Color & Pattern Line) ---")
    env = AzulEnv(num_players=2)
    env.reset()
    
    # Setup: Player 0 has Blue (0) in Pattern Line 0
    env.players[0]['pattern_lines'][0][0] = 0 # Blue
    print(f"Player 0 Pattern Line 0: {env.players[0]['pattern_lines'][0]}")
    
    # Ensure Factory 0 has Red (4)
    env.factories[0] = [0, 0, 0, 0, 4] # 4 Reds
    
    # Get valid actions
    actions = env.get_valid_actions()
    
    # Check if taking Red from Factory 0 to Pattern Line 0 is valid
    # Action: (0, 4, 0)
    illegal_action = (0, 4, 0)
    
    if illegal_action in actions:
        print(f"FAIL: Illegal action {illegal_action} (Mixed Colors) is considered valid!")
    else:
        print("PASS: Illegal action (Mixed Colors) correctly filtered.")
        
    # Also test the Wall Color constraint again
    env.reset()
    env.players[0]['wall'][0][0] = 0 # Blue in Row 0
    env.factories[0] = [1, 0, 0, 0, 0] # Blue
    actions = env.get_valid_actions()
    if (0, 0, 0) in actions:
        print(f"FAIL: Illegal action (Wall Constraint) is considered valid!")
    else:
        print("PASS: Illegal action (Wall Constraint) correctly filtered.")

if __name__ == "__main__":
    test_bug1_first_player_token_persistence()
    test_bug2_impossible_center_counts()
    test_bug3_illegal_actions_wall_color()
