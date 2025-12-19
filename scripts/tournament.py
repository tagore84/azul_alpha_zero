# File: scripts/tournament.py


import os
import sys
import random

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from azul.utils import render_obs
from players.maximilian_times import MaximilianTimes
from players.lillo_expertillo import LilloExpertillo
from players.random_player import RandomPlayer
import argparse
from players.random_plus_player import RandomPlusPlayer
from players.deep_mcts_player import DeepMCTSPlayer 
from players.expert_player import ExpertPlayer
from players.heuristic_min_max_mcts_player import HeuristicMinMaxMCTSPlayer
from players.heuristic_player_v2 import HeuristicPlayerV2
from azul.env import AzulEnv


def play_game(p1, p2):
    
    env = AzulEnv()
    obs = env.reset()
    done = False
    print(f"--- Nueva Partida ---")
    print(f"Jugador 0: {p1.__class__.__name__}")
    print(f"Jugador 1: {p2.__class__.__name__}")
    
    turn = 0
    while not done:
        current_player_idx = obs["current_player"]
        current = p1 if current_player_idx == 0 else p2
        
        #print(f"\nTurno {turn}: Jugador {current_player_idx} ({current.__class__.__name__})")
        # Mostrar estado resumido
        #print(f"  Fábricas: {obs['factories']}")
        #print(f"  Centro: {obs['center']}")
        #print(f"  Muros J0: \n{obs['players'][0]['wall']}")
        #print(f"  Muros J1: \n{obs['players'][1]['wall']}")
        #print(f"  Puntuaciones: J0={obs['players'][0]['score']}, J1={obs['players'][1]['score']}")

        action = current.predict(obs)
        # if predict returned a flat index, convert to action tuple
        if not isinstance(action, tuple):
            action = env.index_to_action(int(action))

        
        obs, _, done, _ = env.step(action)
        turn += 1

    scores = env.get_final_scores()
    print(f"\n--- Fin de Partida ---")
    print(f"Puntuaciones finales: J0={scores[0]}, J1={scores[1]}")
    return scores  # devuelve lista de puntuaciones

def expected_score(rating_a, rating_b):
    return 1 / (1 + 10 ** ((rating_b - rating_a) / 400))

def update_elo(rating, expected, score, k=32):
    return rating + k * (score - expected)

def run_tournament(players, num_games, base_rating=1500, num_passes=20):
    match_results = []
    for i, (name_a, A) in enumerate(players.items()):
        for name_b, B in list(players.items())[i+1:]:
            for _ in range(num_games):
                if random.random() < 0.5:
                    current_name_a, current_name_b = name_b, name_a
                    current_A, current_B = B, A
                else:
                    current_name_a, current_name_b = name_a, name_b
                    current_A, current_B = A, B
                print(f"Jugando {current_name_a} vs {current_name_b} - Partida {_ + 1}/{num_games}")
                scores = play_game(current_A, current_B)
                match_results.append((current_name_a, current_name_b, scores[0], scores[1]))
                print(f"Resultados: {current_name_a} {scores[0]} - {current_name_b} {scores[1]}")
                print("\n")

    
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
        #"AZ_Cycle01": DeepMCTSPlayer("data/checkpoints/model_cycle_1.pt", device="cpu", mcts_iters=50, cpuct=1.0),
        #"AZ500": DeepMCTSPlayer("data/model_history/model_checkpoint_500.pt", device="cpu", mcts_iters=100, cpuct=1),
        #"AZ1014": DeepMCTSPlayer("data/checkpoint_dir_2/model_epoch_019.pt", device="cpu", mcts_iters=100, cpuct=1),
        #"AZ1002": DeepMCTSPlayer("data/model_history/model_epoch_002.pt", device="cpu", mcts_iters=1, cpuct=0),
        #"AZ1000": DeepMCTSPlayer("data/model_history/model_checkpoint_1000.pt", device="cpu", mcts_iters=1, cpuct=0),
        #"MCTS-200": HeuristicMinMaxMCTSPlayer(strategy='mcts', simulations=200),
        "MCTS-10k": HeuristicMinMaxMCTSPlayer(strategy='mcts', simulations=500),
        "MinMax-4": HeuristicMinMaxMCTSPlayer(strategy='minmax', depth=4),
        #"Heuv2": HeuristicPlayerV2(),
        #"Exp": ExpertPlayer(),
        # añade más aquí
    }

    final_ratings = run_tournament(players, num_games=args.n_games, num_passes=args.n_passes)