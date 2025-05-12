# src/train/self_play.py

import os
from concurrent.futures import ThreadPoolExecutor, as_completed
import random
from typing import Any, List, Dict
from azul.env import AzulEnv
from azul.utils import print_wall

def _run_one_game(game_idx: int, env: AzulEnv, model: Any) -> List[Dict[str, Any]]:
    print(f"[Self-play] Worker starting game {game_idx}", flush=True)
    examples = play_game(env.clone())
    return examples

def select_action_no_mcts(env: AzulEnv) -> int:
    """
    Select an action without using MCTS.
    This is a placeholder function and should be replaced with
    the actual action selection logic.
    """
    # For now, just select a random action
    return random.choice(env.get_valid_actions())

def play_game(
    env: AzulEnv
) -> List[Dict[str, Any]]:
    """
    Play one full game with MCTS + model self-play.
    Returns a list of transitions: each is a dict with keys
      - 'obs': flat observation vector (np.ndarray)
      - 'pi':  policy target distribution (np.ndarray)
      - 'v':   value target (+1/-1) for the player to move
    """
    print("[Play-game] Starting new self-play game", flush=True)
    move_idx = 0
    memory = []
    done = False

    while not done:
        print(f"[Play-game] Round {env.round_count} Move {move_idx} - NOT running MCTS", flush=True)
        

        # Select and play action
        action = select_action_no_mcts(env)
        _, reward, done, info = env.step(action)
        move_idx += 1

    print(f"[Play-game] Finished game in {move_idx} moves", flush=True)
    print_wall(env.players[0]['wall'], title="player 0")
    print_wall(env.players[1]['wall'], title="player 1")
    # Determine game winner (highest final score)
    final_obs = env._get_obs()
    scores = [p['score'] for p in final_obs['players']]
    if scores[0] > scores[1]:
        winner = 1
    elif scores[1] > scores[0]:
        winner = 2
    else:
        winner = 0

    # Convert memory to training examples with value targets
    examples = []
    for entry in memory:
        v = 1.0 if entry['player'] == winner else -1.0
        examples.append({
            'obs': entry['obs'],
            'pi': entry['pi'],
            'v': v
        })

    return examples

def generate_self_play_games(
    verbose: bool,
    env: AzulEnv,
    model: Any
) -> List[Dict[str, Any]]:
    """
    Generate multiple self-play games in parallel.
    Returns a flat list of training examples.
    """
    device = next(model.parameters()).device
    if device.type == 'mps':
        print(f"[Self-play] MPS detected ({device}), running games sequentially", flush=True)
        all_examples: List[Dict[str, Any]] = []
        
        examples = _run_one_game(1, env, model)
        all_examples.extend(examples)
        print(f"[Self-play] Completed the game", flush=True)
        
        print(f"[Self-play] Completed", flush=True)
        return all_examples
    else:
        # Determine number of threads
        n_workers = 1
        print(f"[Self-play] Launching {n_workers} parallel workers", flush=True)
        all_examples: List[Dict[str, Any]] = []


        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            futures = {executor.submit(_run_one_game, i+1, env, model): i+1 for i in range(1)}
            for future in as_completed(futures):
                idx = futures[future]
                try:
                    examples = future.result()
                    all_examples.extend(examples)
                    print(f"[Self-play] Completed game {idx}/{1}", flush=True)
                except Exception as e:
                    print(f"[Self-play] Game {idx} failed with error: {e}", flush=True)

        print(f"[Self-play] Completed generation of {1} games", flush=True)
        return all_examples