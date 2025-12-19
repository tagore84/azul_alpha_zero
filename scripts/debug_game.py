
import sys
import os
import numpy as np
import time

# Add project src folder to PYTHONPATH
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from azul.env import AzulEnv
from players.random_plus_player import RandomPlusPlayer
from players.heuristic_min_max_mcts_player import HeuristicMinMaxMCTSPlayer
from players.heuristic_player_v2 import HeuristicPlayerV2
from players.deep_mcts_player import DeepMCTSPlayer

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
    model_path = "data/checkpoints_v5/best.pt"
    p0 = DeepMCTSPlayer(model_path, device="cpu", mcts_iters=1000, cpuct=2)
    p1 = DeepMCTSPlayer(model_path, device="cpu", mcts_iters=1000, cpuct=2)
    #p0 = HeuristicMinMaxMCTSPlayer(strategy='mcts', simulations=1500)
    #p1 = HeuristicMinMaxMCTSPlayer(strategy='minmax', depth=4)
    players = [p0, p1]
    move_times = {0: [], 1: []}
    
    obs = env.reset()
    done = False
    
    log_file = "logs_v5/debug_game_verification.txt"
    print(f"Running debug game... Logging to {log_file}")
    
    with open(log_file, "w") as f:
        f.write("Starting Debug Game: HeuristicPlayerV2 vs HeuristicPlayerV2\n")
        
        while not done:
            current_player_idx = env.current_player
            player = players[current_player_idx]
            
            log_state(env, f)
            
            # Get action
            # HeuristicPlayer always returns tuple
            start_time = time.time()
            action_raw = player.predict(obs)
            end_time = time.time()
            move_duration = end_time - start_time
            move_times[current_player_idx].append(move_duration)
            
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
        
        f.write("\nAverage Move Times:\n")
        for p_idx, times in move_times.items():
            if times:
                avg_time = sum(times) / len(times)
                f.write(f"Player {p_idx}: {avg_time:.4f} seconds\n")
                print(f"Player {p_idx} Average Time: {avg_time:.4f} s")
            else:
                f.write(f"Player {p_idx}: No moves\n")
    
    print("Game finished.")

if __name__ == "__main__":
    main()
