
# src/train/self_play.py

import os
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import numpy as np
import torch
from typing import Any, List, Dict, Tuple
from azul.env import AzulEnv
from mcts.mcts import MCTS

def _run_one_game(game_idx: int, env: AzulEnv, model: Any, simulations: int, cpuct: float, 
                  temp_threshold: int, noise_alpha: float, noise_eps: float):
    #print(f"[Self-play] Worker starting game {game_idx}", flush=True)
    examples, stats = play_game(env.clone(), model, simulations, cpuct, 
                              temperature_threshold=temp_threshold,
                              noise_alpha=noise_alpha,
                              noise_epsilon=noise_eps)
    return examples, stats

def play_game(
    env: AzulEnv,
    model: Any,
    simulations: int = 100,
    cpuct: float = 1.0,
    temperature_threshold: int = 30,
    noise_alpha: float = 0.3,
    noise_epsilon: float = 0.25
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Play one full game with MCTS + model self-play.
    Returns a list of transitions: each is a dict with keys
      - 'obs': flat observation vector (np.ndarray)
      - 'pi':  policy target distribution (np.ndarray)
      - 'v':   value target (normalized score difference) for the player to move
    """
    
    start_time = time.perf_counter()
    move_idx = 0
    mcts = MCTS(env, model=model, simulations=simulations, cpuct=cpuct)
    memory = []
    done = False
    
    # MCTS Statistics tracking
    mcts_stats = {
        'total_visits': [],
        'policy_entropy': [],
        'tree_reuse_count': 0,
        'tree_reset_count': 0
    }

    while not done:
        # Add Dirichlet noise to root for exploration
        if move_idx == 0:  # Only add noise at the start of the search for this move
             mcts.add_root_noise(alpha=noise_alpha, epsilon=noise_epsilon)
        
        # Run MCTS to populate visit counts
        mcts.run()
        root = mcts.root
        
        # Track MCTS statistics
        mcts_stats['total_visits'].append(root.visits)
        
        # Build policy target from visit counts
        visits = np.zeros(env.action_size, dtype=np.float32)
        for action, node in root.children.items():
            idx = env.action_to_index(action)
            visits[idx] = node.visits
        if visits.sum() > 0:
            pi_target = visits / visits.sum()
            # Calculate policy entropy: -sum(p * log(p))
            entropy = -np.sum(pi_target * np.log(pi_target + 1e-10))
            mcts_stats['policy_entropy'].append(entropy)
        else:
            pi_target = visits
            mcts_stats['policy_entropy'].append(0.0)

        # Record current observation and policy target
        obs = env._get_obs()
        obs_flat = env.encode_observation(obs)
        memory.append({
            'obs': obs_flat,
            'pi': pi_target.copy(),
            'player': obs['current_player']
        })
        
        # Select action with temperature
        # Select action with temperature
        # Dynamic Temperature based on round number
        # Rounds 1-2: T=1.0 (Exploration)
        # Rounds 3-4: T=0.5 (Reduced Exploration)
        # Rounds 5+:  T=0.0 (Greedy)
        current_round = env.round_count
        if current_round <= 2:
            temp = 1.0
        elif current_round <= 4:
            temp = 0.5
        else:
            temp = 0.0
        
        # Override with greedy if threshold is explicitly set to 0 (for validation)
        if temperature_threshold == 0:
            temp = 0.0
        action = mcts.select_action(temperature=temp)
        
        # Track tree reuse
        was_in_tree = action in mcts.root.children
        
        _, reward, done, info = env.step(action)
        # Advance MCTS tree (reuse subtree)
        mcts.advance(action, env)
        
        # Update reuse stats
        if was_in_tree:
            mcts_stats['tree_reuse_count'] += 1
        else:
            mcts_stats['tree_reset_count'] += 1
        
        move_idx += 1
    elapsed = time.perf_counter() - start_time
    
    # Log MCTS statistics
    avg_visits = np.mean(mcts_stats['total_visits']) if mcts_stats['total_visits'] else 0
    avg_entropy = np.mean(mcts_stats['policy_entropy']) if mcts_stats['policy_entropy'] else 0
    reuse_rate = mcts_stats['tree_reuse_count'] / (mcts_stats['tree_reuse_count'] + mcts_stats['tree_reset_count']) if move_idx > 0 else 0
    print(f"[Play-game] Finished in {move_idx} moves at {time.strftime('%H:%M:%S')}, time: {elapsed:.2f}s")
    print(f"[Play-game] MCTS: avg_visits={avg_visits:.1f}, avg_entropy={avg_entropy:.2f}, reuse_rate={reuse_rate:.2%}")
    # Calculate score difference
    final_obs = env._get_obs()
    scores = [p['score'] for p in final_obs['players']]
    # Assuming 2 players
    score_p0 = scores[0]
    score_p1 = scores[1]
    
    # Win/Loss Value Target
    # +1 for Win, -1 for Loss, 0 for Draw
    if score_p0 > score_p1:
        diff_0 = 1.0
        diff_1 = -1.0
    elif score_p0 < score_p1:
        diff_0 = -1.0
        diff_1 = 1.0
    else:
        diff_0 = 0.0
        diff_1 = 0.0
    
    # Convert memory to training examples with value targets
    examples = []
    for entry in memory:
        # Value is from perspective of the player who moved
        v = diff_0 if entry['player'] == 0 else diff_1
        examples.append({
            'obs': entry['obs'],
            'pi': entry['pi'],
            'v': v
        })

    # Prepare MCTS statistics summary
    stats_summary = {
        'avg_visits': avg_visits,
        'avg_entropy': avg_entropy,
        'reuse_rate': reuse_rate,
        'move_count': move_idx
    }

    return examples, stats_summary

def generate_self_play_games(
    verbose: bool,
    n_games: int,
    env: AzulEnv,
    model: Any,
    simulations: int = 100,
    cpuct: float = 1.0,
    temperature_threshold: int = 30,
    noise_alpha: float = 0.3,
    noise_epsilon: float = 0.25
):
    """
    Generate multiple self-play games in parallel.
    Returns a tuple of (examples, aggregate_stats).
    """
    from multiprocessing import get_context

    device = next(model.parameters()).device
    global_start = time.time()
    print(f"[Self-play] Estimated total end time will be shown after first game...", flush=True)

    # Aggregate statistics
    all_stats = {
        'avg_visits': [],
        'avg_entropy': [],
        'reuse_rate': [],
        'move_count': []
    }

    if device.type == 'cuda':
        print(f"[Self-play] CUDA GPU detected ({device}), running games sequentially (with GPU reuse)", flush=True)
        all_examples = []
        for i in range(n_games):
            examples, stats = _run_one_game(i+1, env, model, simulations, cpuct, 
                                          temperature_threshold, noise_alpha, noise_epsilon)
            all_examples.extend(examples)
            all_stats['avg_visits'].append(stats['avg_visits'])
            all_stats['avg_entropy'].append(stats['avg_entropy'])
            all_stats['reuse_rate'].append(stats['reuse_rate'])
            all_stats['move_count'].append(stats['move_count'])
            print(f"[Self-play] Completed game {i+1}/{n_games}", flush=True)
            if i == 0:
                elapsed = time.time() - global_start
                estimated_total = elapsed * n_games
                estimated_end = time.localtime(global_start + estimated_total)
                estimated_str = time.strftime('%H:%M:%S', estimated_end)
                print(f"[Self-play] Estimated completion time: {estimated_str}", flush=True)
        print(f"[Self-play] Completed generation of {n_games} games", flush=True)
        
        # Calculate aggregate statistics
        aggregate = {
            'avg_visits': np.mean(all_stats['avg_visits']),
            'avg_entropy': np.mean(all_stats['avg_entropy']),
            'avg_reuse_rate': np.mean(all_stats['reuse_rate']),
            'avg_move_count': np.mean(all_stats['move_count'])
        }
        return all_examples, aggregate

    else:
        # MPS or CPU -> Sequential execution
        # CRITICAL FIX: ThreadPoolExecutor with shared model causes race conditions on BatchNorm
        # running_mean/running_var, corrupting the model. Run sequentially for safety.
        if device.type == 'mps':
            print(f"[Self-play] MPS detected ({device}), moving model to CPU for execution", flush=True)
            model = model.to('cpu')
        
        print(f"[Self-play] Running {n_games} games SEQUENTIALLY (thread-safe)", flush=True)
        all_examples = []
        for i in range(n_games):
            examples, stats = _run_one_game(i+1, env, model, simulations, cpuct, 
                                          temperature_threshold, noise_alpha, noise_epsilon)
            all_examples.extend(examples)
            all_stats['avg_visits'].append(stats['avg_visits'])
            all_stats['avg_entropy'].append(stats['avg_entropy'])
            all_stats['reuse_rate'].append(stats['reuse_rate'])
            all_stats['move_count'].append(stats['move_count'])
            
            if i == 0:
                elapsed = time.time() - global_start
                estimated_total = elapsed * n_games
                estimated_end = time.localtime(global_start + estimated_total)
                estimated_str = time.strftime('%H:%M:%S', estimated_end)
                print(f"[Self-play] Estimated completion time: {estimated_str}", flush=True)
        
        print(f"[Self-play] Completed generation of {n_games} games", flush=True)
        
        # Calculate aggregate statistics
        aggregate = {
            'avg_visits': np.mean(all_stats['avg_visits']) if all_stats['avg_visits'] else 0,
            'avg_entropy': np.mean(all_stats['avg_entropy']) if all_stats['avg_entropy'] else 0,
            'avg_reuse_rate': np.mean(all_stats['reuse_rate']) if all_stats['reuse_rate'] else 0,
            'avg_move_count': np.mean(all_stats['move_count']) if all_stats['move_count'] else 0
        }
        return all_examples, aggregate