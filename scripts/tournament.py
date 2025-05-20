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

def run_tournament(players, num_games, base_rating=1500):
    ratings = {name: base_rating for name in players}
    # Crea un array con 0's por cada jugador
    wins = {name: 0 for name in players}
    # Head-to-head wins matrix: filas=ganador, columnas=perdedor
    results_matrix = {
        name: {opponent: 0 for opponent in players if opponent != name}
        for name in players
    }
    for i, (name_a, A) in enumerate(players.items()):
        for name_b, B in list(players.items())[i+1:]:
            for _ in range(num_games):
                print(f"Jugando {name_a} vs {name_b} - Partida {_ + 1}/{num_games}")
                scores = play_game(A, B)
                # asumimos jugador 0 → A, 1 → B
                if scores[0] > scores[1]:
                    result_a, result_b = 1, 0
                    wins[name_a] += 1
                    # Update head-to-head matrix
                    if result_a == 1:
                        results_matrix[name_a][name_b] += 1
                    elif result_b == 1:
                        results_matrix[name_b][name_a] += 1
                elif scores[0] < scores[1]:
                    result_a, result_b = 0, 1
                    wins[name_b] += 1
                    # Update head-to-head matrix
                    if result_a == 1:
                        results_matrix[name_a][name_b] += 1
                    elif result_b == 1:
                        results_matrix[name_b][name_a] += 1
                else:
                    result_a = result_b = 0.5
                ea = expected_score(ratings[name_a], ratings[name_b])
                eb = expected_score(ratings[name_b], ratings[name_a])
                ratings[name_a] = update_elo(ratings[name_a], ea, result_a)
                ratings[name_b] = update_elo(ratings[name_b], eb, result_b)
                print(f"Resultados: {name_a} {scores[0]} - {name_b} {scores[1]}")
                print("\n")
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
    parser.add_argument("--games", type=int, default=5, help="Partidas por enfrentamiento")
    args = parser.parse_args()

    players = {
        #"Heu": HeuristicPlayer(),
        "Alfa100": DeepMCTSPlayer("data/checkpoint_dir/checkpoint_latest_mac.pt", device="cpu", mcts_iters=50, cpuct=1.0),
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

    final_ratings = run_tournament(players, num_games=args.games)