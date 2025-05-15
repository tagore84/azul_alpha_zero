# File: scripts/tournament.py


import os
import sys


sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
import argparse
from collections import defaultdict
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
    return env.get_final_scores()  # devuelve lista de puntuaciones

def expected_score(rating_a, rating_b):
    return 1 / (1 + 10 ** ((rating_b - rating_a) / 400))

def update_elo(rating, expected, score, k=32):
    return rating + k * (score - expected)

def run_tournament(players, num_games, base_rating=1500):
    ratings = {name: base_rating for name in players}
    wins = defaultdict(int)

    for i, (name_a, A) in enumerate(players.items()):
        for name_b, B in list(players.items())[i+1:]:
            for _ in range(num_games):
                scores = play_game(A, B)
                # asumimos jugador 0 → A, 1 → B
                if scores[0] > scores[1]:
                    result_a, result_b = 1, 0
                elif scores[0] < scores[1]:
                    result_a, result_b = 0, 1
                else:
                    result_a = result_b = 0.5
                ea = expected_score(ratings[name_a], ratings[name_b])
                eb = expected_score(ratings[name_b], ratings[name_a])
                ratings[name_a] = update_elo(ratings[name_a], ea, result_a)
                ratings[name_b] = update_elo(ratings[name_b], eb, result_b)

    return ratings

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--games", type=int, default=20, help="Partidas por enfrentamiento")
    args = parser.parse_args()

    players = {
        "Heuristic_1": HeuristicPlayer(),
        "Heuristic_2": HeuristicPlayer(),
        "DeepMCTSPlayer": DeepMCTSPlayer("data/checkpoint_dir/model_epoch_019.pt", device="cpu", mcts_iters=5, cpuct=2.0,)
        # añade más aquí
    }

    final_ratings = run_tournament(players, num_games=args.games)
    for name, rating in final_ratings.items():
        print(f"{name}: {rating:.1f}")