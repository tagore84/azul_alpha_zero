import sys
import os
from datetime import datetime
# Add project src folder to PYTHONPATH
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from azul.env import AzulEnv
from players.heuristic_player import HeuristicPlayer 

def run_game():
    game_num = 1
    while True:
        print(f"\n--- Game {game_num} ---")
        env = AzulEnv(num_players=2)
        obs = env.reset()
        players = [HeuristicPlayer(), HeuristicPlayer()]
        done = False

        while not done:
            current_player = env.current_player
            action = players[current_player].predict(obs)
            obs, reward, done, info = env.step(env.index_to_action(action))

        print("Game finished.")
        print("Final scores:", env.get_final_scores())
        game_num += 1

if __name__ == "__main__":
    run_game()