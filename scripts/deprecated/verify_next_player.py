
import numpy as np
from azul.env import AzulEnv

def test_next_player_logic():
    print("\n--- Testing Next Player Logic ---")
    env = AzulEnv(num_players=2)
    env.reset()
    
    # Force Factory 0 to have Color 0 and Color 1
    env.factories[0] = [1, 1, 0, 0, 0] # 1 Blue, 1 Yellow
    
    # Round 1.
    # P0 moves. Takes from factory 0 (Blue).
    env.step((0, 0, 0))
    
    # P1 moves. Takes from Center (which has residues).
    # P1 should get token.
    # Center should have tiles because P0 took from factory.
    # Let's ensure center has tiles.
    # Factory 0 had 4 tiles. P0 took some. Others went to center.
    print(f"Center after P0 move: {env.center}")
    
    # P1 takes from Center.
    # We need to find a color in center.
    color_in_center = -1
    for c in range(5):
        if env.center[c] > 0:
            color_in_center = c
            break
            
    if color_in_center == -1:
        print("Error: Center empty?")
        return

    print(f"P1 taking color {color_in_center} from center.")
    env.step((5, color_in_center, 0)) # Source 5 is center.
    
    # Check if P1 has token (5)
    if 5 in env.players[1]['floor_line']:
        print("P1 has token 5.")
    else:
        print("P1 does NOT have token 5. (Floor full?)")
        
    # Check first_player_next_round
    print(f"first_player_next_round: {env.first_player_next_round}")
    
    if env.first_player_next_round == 1:
        print("PASS: first_player_next_round is 1.")
    else:
        print(f"FAIL: first_player_next_round is {env.first_player_next_round}, expected 1.")
        
    # Finish round.
    # We can force finish by clearing factories/center.
    env.factories[:] = 0
    env.center[:] = 0
    
    # Call step to trigger end round?
    # No, step needs valid action.
    # We can call _end_round directly?
    # But we want to test step logic.
    # But we can't step if no actions.
    # If no actions, step raises error.
    # But in game loop, we check is_round_over.
    
    print("Forcing end of round via _end_round...")
    env._end_round()
    
    print(f"Current player for Round 2: {env.current_player}")
    
    if env.current_player == 1:
        print("PASS: P1 starts Round 2.")
    else:
        print(f"FAIL: P{env.current_player} starts Round 2, expected P1.")

if __name__ == "__main__":
    test_next_player_logic()
