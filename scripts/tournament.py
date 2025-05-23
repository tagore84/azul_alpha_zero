# File: scripts/tournament.py


import os
import sys


sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from azul.utils import render_obs
from players.maximilian_times import MaximilianTimes
from players.lillo_expertillo import LilloExpertillo
from players.random_player import RandomPlayer
import argparse
from players.deep_mcts_player import DeepMCTSPlayer 
from players.expert_player import ExpertPlayer
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

def run_tournament(players, num_games, base_rating=1500, num_passes=20):
    match_results = []
    for i, (name_a, A) in enumerate(players.items()):
        for name_b, B in list(players.items())[i+1:]:
            for _ in range(num_games):
                print(f"Jugando {name_a} vs {name_b} - Partida {_ + 1}/{num_games}")
                scores = play_game(A, B)
                match_results.append((name_a, name_b, scores[0], scores[1]))
                print(f"Resultados: {name_a} {scores[0]} - {name_b} {scores[1]}")
                print("\n")

    import random
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

    print("Resultados finales:")
    sorted_ratings = sorted(ratings.items(), key=lambda x: x[1], reverse=True)
    for i, (name, rating) in enumerate(sorted_ratings):
        print(f"{i + 1}. {name}: {rating:.1f} ({wins[name]} victorias)")
    # Print head-to-head wins matrix
    print("\nMatriz de victorias (filas=ganador, columnas=perdedor):")
    names = list(players.keys())
    # Header row
    print("\t" + "\t".join(names))
    for winner in names:
        row = [str(results_matrix[winner].get(loser, 0)) for loser in names]
        print(f"{winner}\t" + "\t".join(row))
    return {name: rating for name, rating in sorted_ratings}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_games", type=int, default=5, help="Partidas por enfrentamiento")
    parser.add_argument("--n_passes", type=int, default=20, help="Número de pasadas para cálculo de Elo")
    args = parser.parse_args()

    players = {
        #"Heu": HeuristicPlayer(),
        #"Alfa200": DeepMCTSPlayer("data/model_history/model_checkpoint_200.pt", device="cpu", mcts_iters=1, cpuct=0),
        #"Alfa300": DeepMCTSPlayer("data/model_history/model_checkpoint_300.pt", device="cpu", mcts_iters=1, cpuct=0),
        "Alfa40g": DeepMCTSPlayer("data/model_history/model_checkpoint_400_g.pt", device="cpu", mcts_iters=1, cpuct=0),
        "Alfa40m": DeepMCTSPlayer("data/model_history/model_checkpoint_400.pt", device="cpu", mcts_iters=1, cpuct=0),
        "Alfa4gA": DeepMCTSPlayer("data/model_history/model_checkpoint_g_50_a.pt", device="cpu", mcts_iters=1, cpuct=0),
        #"Alfa240": DeepMCTSPlayer("data/checkpoint_dir/model_epoch_040_mac.pt", device="cpu", mcts_iters=1, cpuct=0),
        #"A100M": DeepMCTSPlayer("data/checkpoint_dir/checkpoint_latest_mac_100.pt", device="cpu", mcts_iters=5, cpuct=0.1),
        #"A100R": DeepMCTSPlayer("data/checkpoint_dir/checkpoint_latest_mac_100.pt", device="cpu", mcts_iters=10, cpuct=3.0),
        #"A200B": DeepMCTSPlayer("data/checkpoint_dir/checkpoint_latest_mac.pt", device="cpu", mcts_iters=50, cpuct=0.5),
        #"A200M": DeepMCTSPlayer("data/checkpoint_dir/checkpoint_latest_mac.pt", device="cpu", mcts_iters=5, cpuct=0.1),
        #"A200R": DeepMCTSPlayer("data/checkpoint_dir/checkpoint_latest_mac.pt", device="cpu", mcts_iters=10, cpuct=3.0),
        "Exp": ExpertPlayer(),
        #"Exp2": ExpertPlayer(),
        "Rand": RandomPlayer(),
        #"Lillo1": LilloExpertillo(),
        #"Lillo2": LilloExpertillo(),
        #"Maxi": MaximilianTimes(5, 1, 1.2, 1.0),
        # añade más aquí
    }

    final_ratings = run_tournament(players, num_games=args.n_games, num_passes=args.n_passes)