
import sys
import os
import numpy as np

# Add project src folder to PYTHONPATH
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from azul.env import AzulEnv
from players.deep_mcts_player import DeepMCTSPlayer
from players.heuristic_player import HeuristicPlayer

def log_state(env, f):
    f.write(f"\n{'='*20} Round {env.round_count} | Player {env.current_player} to move {'='*20}\n")
    f.write(f"Factories:\n{env.factories}\n")
    f.write(f"Center: {env.center} (First token: {env.first_player_token})\n")
    
    for idx, p in enumerate(env.players):
        f.write(f"Player {idx} Score: {p['score']}\n")
        f.write(f"  Wall:\n{p['wall']}\n")
        f.write(f"  Pattern Lines:\n")
        for line in p['pattern_lines']:
            f.write(f"    {line}\n")
        f.write(f"  Floor Line: {p['floor_line']}\n")

def main():
    env = AzulEnv()
    model_path = "data/checkpoints/best.pt"
    p1 = HeuristicPlayer()
    p2 = HeuristicPlayer()
    players = [p1, p2]
    
    obs = env.reset()
    done = False
    
    log_file = "logs/debug_game_verification.txt"
    print(f"Running debug game... Logging to {log_file}")
    
    with open(log_file, "w") as f:
        f.write("Starting Debug Game: HeuristicPlayer vs HeuristicPlayer\n")
        
        while not done:
            current_player_idx = env.current_player
            player = players[current_player_idx]
            
            log_state(env, f)
            
            # Get action
            # HeuristicPlayer always returns tuple
            action_raw = player.predict(obs)
            
            if isinstance(action_raw, (int, np.integer)):
                 action = env.index_to_action(int(action_raw))
            else:
                 action = action_raw
            
            f.write(f"\n>>> Action chosen: {action} (Source: {action[0]}, Color: {action[1]}, Dest: {action[2]})\n")
            
            obs, reward, done, info = env.step(action)
            
        f.write("\n" + "="*30 + "\n")
        f.write("=== GAME OVER ===\n")
        f.write("="*30 + "\n")
        log_state(env, f)
        winners = env.get_winner()
        f.write(f"Winners: Player {winners}\n")
        f.write(f"Final Scores: {[p['score'] for p in env.players]}\n")
    
    print("Game finished.")

if __name__ == "__main__":
    main()
