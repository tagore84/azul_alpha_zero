
import os
import sys
import random
import argparse

# Add src to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from azul.utils import render_obs
from players.random_player import RandomPlayer
from players.deep_mcts_player import DeepMCTSPlayer 
from players.heuristic_player import HeuristicPlayer
from azul.env import AzulEnv

def play_game(p1, p2):
    env = AzulEnv()
    obs = env.reset()
    done = False
    while not done:
        current = p1 if obs["current_player"] == 0 else p2
        action = current.predict(obs)
        # if predict returned a flat index, convert to action tuple
        if not isinstance(action, tuple):
            action = env.index_to_action(int(action))
        obs, _, done, _ = env.step(action)
    return env.get_final_scores()

def expected_score(rating_a, rating_b):
    return 1 / (1 + 10 ** ((rating_b - rating_a) / 400))

def update_elo(rating, expected, score, k=32):
    return rating + k * (score - expected)

def run_tournament(players, num_games, base_rating=1500, num_passes=20):
    match_results = []
    # Play matches
    player_names = list(players.keys())
    for i, name_a in enumerate(player_names):
        for name_b in player_names[i+1:]:
            player_a = players[name_a]
            player_b = players[name_b]
            
            for g in range(num_games):
                # Swap starting player
                if random.random() < 0.5:
                    p1_name, p2_name = name_b, name_a
                    p1, p2 = player_b, player_a
                else:
                    p1_name, p2_name = name_a, name_b
                    p1, p2 = player_a, player_b
                
                print(f"Playing {p1_name} vs {p2_name} - Game {g + 1}/{num_games}")
                scores = play_game(p1, p2)
                match_results.append((p1_name, p2_name, scores[0], scores[1]))
                print(f"Result: {p1_name} {scores[0]} - {p2_name} {scores[1]}")

    # Calculate Elo
    ratings = {name: base_rating for name in players}
    for _ in range(num_passes):
        random.shuffle(match_results)
        for name_a, name_b, score_a, score_b in match_results:
            if score_a > score_b:
                result_a, result_b = 1, 0
            elif score_a < score_b:
                result_a, result_b = 0, 1
            else:
                result_a = result_b = 0.5
            
            ea = expected_score(ratings[name_a], ratings[name_b])
            eb = expected_score(ratings[name_b], ratings[name_a])
            ratings[name_a] = update_elo(ratings[name_a], ea, result_a)
            ratings[name_b] = update_elo(ratings[name_b], eb, result_b)

    # Calculate wins/losses matrix
    wins = {name: 0 for name in players}
    results_matrix = {
        name: {opponent: 0 for opponent in players if opponent != name}
        for name in players
    }
    for name_a, name_b, score_a, score_b in match_results:
        if score_a > score_b:
            wins[name_a] += 1
            results_matrix[name_a][name_b] += 1
        elif score_b > score_a:
            wins[name_b] += 1
            results_matrix[name_b][name_a] += 1

    print("\nFinal Results:")
    sorted_ratings = sorted(ratings.items(), key=lambda x: x[1], reverse=True)
    for i, (name, rating) in enumerate(sorted_ratings):
        print(f"{i + 1}. {name}: {rating:.1f} ({wins[name]} wins)")

    print("\nWin Matrix (row=winner, col=loser):")
    print("\t" + "\t".join(player_names))
    for winner in player_names:
        row = [str(results_matrix[winner].get(loser, 0)) if winner != loser else "-" for loser in player_names]
        print(f"{winner}\t" + "\t".join(row))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_games", type=int, default=10, help="Games per matchup")
    args = parser.parse_args()

    model_path = "data/checkpoints/model_cycle_11.pt"
    if not os.path.exists(model_path):
        print(f"Warning: {model_path} not found. Please check the path.")
        # Fallback or exit? I'll let it crash if DeepMCTSPlayer fails, but the print helps.

    players = {
        "Cycle11": DeepMCTSPlayer(model_path, device="cpu", mcts_iters=50, cpuct=1.0),
        "Heuristic": HeuristicPlayer(),
        "Random": RandomPlayer(),
    }

    print(f"Starting tournament with players: {list(players.keys())}")
    run_tournament(players, num_games=args.n_games)
