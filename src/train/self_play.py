# src/train/self_play.py

import os
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import numpy as np
from typing import Any, List, Dict
from azul.env import AzulEnv
from mcts.mcts import MCTS

def _run_one_game(game_idx: int, env: AzulEnv, model: Any, simulations: int, cpuct: float) -> List[Dict[str, Any]]:
    #print(f"[Self-play] Worker starting game {game_idx}", flush=True)
    examples = play_game(env.clone(), model, simulations, cpuct)
    return examples

def play_game(
    env: AzulEnv,
    model: Any,
    simulations: int = 100,
    cpuct: float = 1.0
) -> List[Dict[str, Any]]:
    """
    Play one full game with MCTS + model self-play.
    Returns a list of transitions: each is a dict with keys
      - 'obs': flat observation vector (np.ndarray)
      - 'pi':  policy target distribution (np.ndarray)
      - 'v':   value target (+1/-1) for the player to move
    """
    
    start_time = time.perf_counter()
    move_idx = 0
    mcts = MCTS(env, model=model, simulations=simulations, cpuct=cpuct)
    memory = []
    done = False

    while not done:
        # Run MCTS to populate visit counts
        mcts.run()
        root = mcts.root
        # Build policy target from visit counts
        visits = np.zeros(env.action_size, dtype=np.float32)
        for action, node in root.children.items():
            idx = env.action_to_index(action)
            visits[idx] = node.visits
        if visits.sum() > 0:
            pi_target = visits / visits.sum()
        else:
            pi_target = visits

        # Record current observation and policy target
        obs = env._get_obs()
        obs_flat = env.encode_observation(obs)
        memory.append({
            'obs': obs_flat,
            'pi': pi_target.copy(),
            'player': obs['current_player']
        })
        # Select and play action
        action = mcts.select_action()
        _, reward, done, info = env.step(action)
        # Advance MCTS tree
        mcts.advance(env)
        move_idx += 1
    elapsed = time.perf_counter() - start_time
    print(f"[Play-game] Finished game in {move_idx} moves at {time.strftime('%H:%M:%S')}, time: {elapsed:.2f}s", flush=True)
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
    n_games: int,
    env: AzulEnv,
    model: Any,
    simulations: int = 100,
    cpuct: float = 1.0
) -> List[Dict[str, Any]]:
    """
    Generate multiple self-play games in parallel.
    Returns a flat list of training examples.
    """
    device = next(model.parameters()).device
    global_start = time.time()
    print(f"[Self-play] Estimated total end time will be shown after first game...", flush=True)
    if device.type == 'mps':
        print(f"[Self-play] MPS detected ({device}), running games sequentially", flush=True)
        all_examples: List[Dict[str, Any]] = []
        for i in range(n_games):
            examples = _run_one_game(i+1, env, model, simulations, cpuct)
            all_examples.extend(examples)
            print(f"[Self-play] Completed game {i+1}/{n_games}", flush=True)
            if True:
                elapsed = time.time() - global_start
                estimated_total = elapsed / (i+1) * n_games
                estimated_end = time.localtime(global_start + estimated_total)
                estimated_str = time.strftime('%H:%M:%S', estimated_end)
                print(f"[Self-play] Estimated completion time: {estimated_str}", flush=True)
        print(f"[Self-play] Completed generation of {n_games} games", flush=True)
        return all_examples
    else:
        print(f"[Self-play] Starting generation of {n_games} games", flush=True)
        # Determine number of threads
        n_workers = min(32, os.cpu_count() or 1)
        print(f"[Self-play] Launching {n_workers} parallel workers", flush=True)

        all_examples: List[Dict[str, Any]] = []

        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            futures = {executor.submit(_run_one_game, i+1, env, model, simulations, cpuct): i+1 for i in range(n_games)}
            completed_games = 0
            for future in as_completed(futures):
                idx = futures[future]
                try:
                    examples = future.result()
                    all_examples.extend(examples)
                    completed_games += 1
                    print(f"[Self-play] Completed game {idx}/{n_games}", flush=True)
                    if completed_games == 1:
                        elapsed = time.time() - global_start
                        estimated_total = elapsed / completed_games * n_games
                        estimated_end = time.localtime(global_start + estimated_total)
                        estimated_str = time.strftime('%H:%M:%S', estimated_end)
                        print(f"[Self-play] Estimated completion time: {estimated_str}", flush=True)
                except Exception as e:
                    print(f"[Self-play] Game {idx} failed with error: {e}", flush=True)

        print(f"[Self-play] Completed generation of {n_games} games", flush=True)
        return all_examples