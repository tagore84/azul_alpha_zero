
# src/train/self_play.py

import os
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import logging
import numpy as np
import torch
from typing import Any, List, Dict, Tuple
from azul.env import AzulEnv
from mcts.mcts import MCTS
from players.heuristic_player import HeuristicPlayer
from players.random_player import RandomPlayer

logger = logging.getLogger(__name__)

def _run_one_game(game_idx: int, env: AzulEnv, model: Any, simulations: int, cpuct: float, 
                  temp_threshold: int, noise_alpha: float, noise_eps: float):
    #print(f"[Self-play] Worker starting game {game_idx}", flush=True)
    examples, stats = play_game(env.clone(), model, simulations, cpuct, 
                              temperature_threshold=temp_threshold,
                              noise_alpha=noise_alpha,
                              noise_epsilon=noise_eps)
    return examples, stats

def _run_one_game_vs_opponent(game_idx: int, env: AzulEnv, model: Any, simulations: int, cpuct: float, 
                  temp_threshold: int, noise_alpha: float, noise_eps: float, opponent_type: str):
    examples, stats = play_game_vs_opponent(env.clone(), model, simulations, cpuct, 
                              temperature_threshold=temp_threshold,
                              noise_alpha=noise_alpha,
                              noise_epsilon=noise_eps,
                              opponent_type=opponent_type)
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
    # Enable Single Player Mode for "Maximize Own Score" logic
    # This prevents the agent from minimizing opponent score in zero-sum backprop
    mcts = MCTS(env, model=model, simulations=simulations, cpuct=cpuct)
    mcts.single_player_mode = False 
    memory = []
    done = False
    
    # MCTS Statistics tracking
    mcts_stats = {
        'total_visits': [],
        'policy_entropy': [],
        'tree_reuse_count': 0,
        'tree_reset_count': 0
    }
    
    # NEW: Track floor moves for Reward Shaping (Option 2)
    p_floor_moves = {0: 0, 1: 0}

    while not done:
        # Add Dirichlet noise to root for exploration
        # Add Dirichlet noise to root for exploration
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
        action_mask = env.get_action_mask()
        memory.append({
            'obs': obs_flat,
            'mask': action_mask,
            'pi': pi_target.copy(),
            'player': obs['current_player']
        })

        
        # Select action with temperature
        current_round = env.round_count
        if move_idx < temperature_threshold:
            temp = 1.0
        else:
            temp = 0.0

        # Override with greedy if threshold is explicitly set to 0 (for validation)
        if temperature_threshold == 0:
            temp = 0.0
        action = mcts.select_action(temperature=temp)
        
        # Track tree reuse
        was_in_tree = action in mcts.root.children
        
        # NEW: Track floor move
        if action[2] == 5: # dest == 5 is floor
            p_floor_moves[env.current_player] += 1
        
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
    logger.info(f"[Play-game] Finished in {move_idx} moves ({current_round} rounds) at {time.strftime('%H:%M:%S')}, time: {elapsed:.2f}s")
    logger.info(f"[Play-game] MCTS: avg_visits={avg_visits:.1f}, avg_entropy={avg_entropy:.2f}, reuse_rate={reuse_rate:.2%}")
    # Calculate score difference
    final_obs = env._get_obs()
    scores = [p['score'] for p in final_obs['players']]
    # Assuming 2 players
    score_p0 = scores[0]
    score_p1 = scores[1]
    
    # Win/Loss Value Target - RELATIVE SCORING (ZERO-SUM)
    # We want to maximize the score difference (MyScore - OppScore)
    # We normalize by 100.0 (reasonable max score difference) and clamp to [-1, 1]
    
    score_diff_0 = score_p0 - score_p1
    score_diff_1 = score_p1 - score_p0

    val_0 = np.clip(score_diff_0 / 100.0, -1.0, 1.0)
    val_1 = np.clip(score_diff_1 / 100.0, -1.0, 1.0)

    # For zero-sum compatibility in MCTS if needed? 
    # MCTS expects a value for the player.
    # We will just give the normalized score as the value.
    # Note: This is NOT zero-sum anymore.
    
    diff_0 = val_0
    diff_1 = val_1



    
    # Helper to calculate penalty (duplicate logic from rules to avoid importing rules here if possible, or just use simple logic)
    # Actually env object doesn't have it exposed publically in my previous memory? 
    # Let's check env.py. It imports calculate_floor_penalization.
    # But I can't access it easily without importing rules.
    # I'll just monkey patch a helper or do it manually.
    # Penalties: -1, -1, -2, -2, -2, -3, -3
    penalties = [-1, -1, -2, -2, -2, -3, -3]
    def calc_pen(fl):
        s = 0
        for i, t in enumerate(fl):
            if t != -1: s += penalties[i]
        return s
    
    # Actually, I'll add a helper alias to env class to avoid strict dependency on 'rules'
    # No, I can't modify env class here.
    # I'll just use a local lambda.
    # But wait, I'm inside play_game module. I can import rules?
    # No, cyclic imports? No.
    # Let's simple define it here.
    penalties_vals = [-1, -1, -2, -2, -2, -3, -3]
    env.calculate_floor_calculation_simple = lambda fl: sum(penalties_vals[i] for i, t in enumerate(fl) if t != -1)
    
    # Convert memory to training examples with value targets
    examples = []
    for entry in memory:
        # Value is from perspective of the player who moved
        v = diff_0 if entry['player'] == 0 else diff_1
        
        # Check for NaN consistency
        if np.isnan(entry['obs']).any() or np.isnan(entry['pi']).any() or np.isnan(v):
            logger.error(f"[Play-game] FATAL: NaN detected in example generation! v={v}, pi_sum={entry['pi'].sum()}")
            continue # Skip bad example
            
        examples.append({
            'obs': entry['obs'],
            'mask': entry['mask'],
            'pi': entry['pi'],
            'v': v
        })


    # NEW: Discard data if max_rounds_reached
    if getattr(env, 'termination_reason', 'normal_end') == "max_rounds_reached":
        logger.warning(f"[Self-play] Game discarded due to max_rounds_reached (Rounds: {current_round})")
        examples = [] # Discard ALL data for this game

    # Prepare MCTS statistics summary
    stats_summary = {
        'avg_visits': avg_visits,
        'avg_entropy': avg_entropy,
        'reuse_rate': reuse_rate,
        'move_count': move_idx,
        # Calculate floor penalties for both players
        'p0_floor_penalty': env.calculate_floor_calculation_simple(final_obs['players'][0]['floor_line']),
        'p1_floor_penalty': env.calculate_floor_calculation_simple(final_obs['players'][1]['floor_line']),
        # Game result info
        'p0_score': score_p0,
        'p1_score': score_p1,
        'round_count': current_round,
        'winner': 0 if score_p0 > score_p1 else (1 if score_p1 > score_p0 else -1),  # -1 = draw
        'termination_reason': getattr(env, 'termination_reason', 'normal_end')
    }

    return examples, stats_summary

def play_game_vs_opponent(
    env: AzulEnv,
    model: Any,
    simulations: int = 100,
    cpuct: float = 1.0,
    temperature_threshold: int = 30,
    noise_alpha: float = 0.3,
    noise_epsilon: float = 0.25,
    opponent_type: str = 'heuristic'
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Play one full game: Agent (MCTS) vs Heuristic or Random.
    Randomly assigns Agent to P0 or P1.
    """
    start_time = time.perf_counter()
    move_idx = 0
    mcts = MCTS(env, model=model, simulations=simulations, cpuct=cpuct)
    mcts.single_player_mode = False # Use zero-sum logic (maximize relative score)
    
    if opponent_type == 'heuristic':
        opponent = HeuristicPlayer()
    elif opponent_type == 'random':
        opponent = RandomPlayer()
    else:
        raise ValueError(f"Unknown opponent type: {opponent_type}")
    memory = []
    done = False
    
    # 50% chance for agent to be P0
    agent_player = np.random.randint(2) 
    
    mcts_stats = {
        'total_visits': [],
        'policy_entropy': [],
        'tree_reuse_count': 0,
        'tree_reset_count': 0
    }

    while not done:
        is_agent_turn = (env.current_player == agent_player)
        
        if is_agent_turn:
            # --- Agent Turn (MCTS) ---
            mcts.add_root_noise(alpha=noise_alpha, epsilon=noise_epsilon)
            mcts.run()
            root = mcts.root
            
            mcts_stats['total_visits'].append(root.visits)
            
            # Policy Target
            visits = np.zeros(env.action_size, dtype=np.float32)
            for action, node in root.children.items():
                idx = env.action_to_index(action)
                visits[idx] = node.visits
            if visits.sum() > 0:
                pi_target = visits / visits.sum()
                entropy = -np.sum(pi_target * np.log(pi_target + 1e-10))
                mcts_stats['policy_entropy'].append(entropy)
            else:
                pi_target = visits
                mcts_stats['policy_entropy'].append(0.0)

            # Store experience
            obs = env._get_obs()
            obs_flat = env.encode_observation(obs)
            action_mask = env.get_action_mask()
            memory.append({
                'obs': obs_flat,
                'mask': action_mask,
                'pi': pi_target.copy(),
                'player': obs['current_player']
            })

            
            # Select Action
            if move_idx < temperature_threshold:
                temp = 1.0
            else:
                temp = 0.0
            if temperature_threshold == 0: temp = 0.0
                
            action = mcts.select_action(temperature=temp)
            was_in_tree = action in mcts.root.children
            if was_in_tree:
                mcts_stats['tree_reuse_count'] += 1
            else:
                mcts_stats['tree_reset_count'] += 1

        else:
            # --- Heuristic Turn ---
            # No MCTS for the heuristic itself, just direct prediction
            # But we might want to store this as a training example too?
            # User plan said: "Store for both, but the heuristic's moves are demonstrations"
            # However, Heuristic doesn't give a policy distribution, just a hard action.
            # We can create a one-hot policy target.
            
            obs = env._get_obs()
            obs_flat = env.encode_observation(obs)
            
            try:
                action_idx = opponent.predict(obs)
                if isinstance(action_idx, tuple): # Should assume index but let's be safe
                     # HeuristicPlayer usually returns index if flat, but check implementation
                     # Reading `heuristic_player.py`: returns best_action (int)
                     action = env.index_to_action(action_idx)
                else: 
                     action = env.index_to_action(action_idx)
            except Exception as e:
                # Fallback to random valid if heuristic fails
                action = env.index_to_action(np.random.choice(env.get_valid_actions()))

            # Create One-Hot Policy
            pi_target = np.zeros(env.action_size, dtype=np.float32)
            pi_target[env.action_to_index(action)] = 1.0
            
            action_mask = env.get_action_mask()
            memory.append({
                'obs': obs_flat,
                'mask': action_mask,
                'pi': pi_target,
                'player': obs['current_player']
            })

            
            # Advance MCTS to keep it in sync (observation update)
            # We don't run MCTS search here, just advance state
            pass

        # Apply action
        _, reward, done, info = env.step(action)
        mcts.advance(action, env)
        move_idx += 1

    elapsed = time.perf_counter() - start_time
    
    # Calculate Results
    final_obs = env._get_obs()
    score_p0 = final_obs['players'][0]['score']
    score_p1 = final_obs['players'][1]['score']
    
    # Value Target: Normalized Relative Score
    # v = clamp((score_self - score_opp) / 100.0, -1, 1)
    
    score_diff_0 = score_p0 - score_p1
    score_diff_1 = score_p1 - score_p0
    
    val_0 = np.clip(score_diff_0 / 100.0, -1.0, 1.0)
    val_1 = np.clip(score_diff_1 / 100.0, -1.0, 1.0)
    
    diff_0 = val_0
    diff_1 = val_1

    # Build examples
    # Build examples
    examples = []
    for entry in memory:
        v = diff_0 if entry['player'] == 0 else diff_1
        
        examples.append({
            'obs': entry['obs'],
            'pi': entry['pi'],
            'v': v
        })

    # NEW: Discard data if max_rounds_reached (Vs Heuristic)
    term_reason = getattr(env, 'termination_reason', 'normal_end')
    if term_reason == "max_rounds_reached":
        logger.warning(f"[Self-play] Game vs {opponent_type} discarded due to max_rounds_reached (Rounds: {env.round_count})")
        examples = [] # Discard data

    # Stats
    avg_visits = np.mean(mcts_stats['total_visits']) if mcts_stats['total_visits'] else 0
    avg_entropy = np.mean(mcts_stats['policy_entropy']) if mcts_stats['policy_entropy'] else 0
    reuse_rate = mcts_stats['tree_reuse_count'] / (mcts_stats['tree_reuse_count'] + mcts_stats['tree_reset_count']) if (mcts_stats['tree_reuse_count'] + mcts_stats['tree_reset_count']) > 0 else 0

    stats_summary = {
        'avg_visits': avg_visits,
        'avg_entropy': avg_entropy,
        'reuse_rate': reuse_rate,
        'move_count': move_idx,
        'p0_score': score_p0,
        'p1_score': score_p1,
        'round_count': env.round_count,
        'winner': 0 if score_p0 > score_p1 else (1 if score_p1 > score_p0 else -1),
        'termination_reason': getattr(env, 'termination_reason', 'normal_end'),
        'opponent': opponent_type,
        'agent_player': agent_player,
        'winner_name': 'Agent' if (score_p0 > score_p1 and agent_player == 0) or (score_p1 > score_p0 and agent_player == 1) else (opponent_type.capitalize() if (score_p0 > score_p1 and agent_player == 1) or (score_p1 > score_p0 and agent_player == 0) else 'Draw')
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
    noise_epsilon: float = 0.25,
    opponent_type: str = 'self',  # 'self' or 'heuristic'
    game_logger: Any = None  # Optional logger for game results
):
    """
    Generate multiple self-play games in parallel.
    Returns a tuple of (examples, aggregate_stats).
    """
    from multiprocessing import get_context

    device = next(model.parameters()).device
    global_start = time.time()
    logger.info(f"[Self-play] Estimated total end time will be shown after first game...")

    # Aggregate statistics
    all_stats = {
        'avg_visits': [],
        'avg_entropy': [],
        'reuse_rate': [],
        'move_count': [],
        'p0_floor_penalty': [],
        'p0_score': [],
        'p1_score': [],
        'round_count': []
    }

    if device.type == 'cuda':
        logger.info(f"[Self-play] CUDA GPU detected ({device}), running games sequentially (with GPU reuse)")
        all_examples = []
        for i in range(n_games):
            if opponent_type != 'self':
                 examples, stats = _run_one_game_vs_opponent(i+1, env, model, simulations, cpuct, 
                                          temperature_threshold, noise_alpha, noise_epsilon, opponent_type)
            else:
                 examples, stats = _run_one_game(i+1, env, model, simulations, cpuct, 
                                          temperature_threshold, noise_alpha, noise_epsilon)
            all_examples.extend(examples)
            all_stats['avg_visits'].append(stats['avg_visits'])
            all_stats['avg_entropy'].append(stats['avg_entropy'])
            all_stats['reuse_rate'].append(stats['reuse_rate'])
            all_stats['move_count'].append(stats['move_count'])
            all_stats['p0_score'].append(stats.get('p0_score', 0))
            all_stats['p1_score'].append(stats.get('p1_score', 0))
            all_stats['round_count'].append(stats.get('round_count', 0))
            
            # Log game result
            winner_str = f"P{stats.get('winner', -1)}" if stats.get('winner', -1) != -1 else "DRAW"
            if 'winner_name' in stats:
                winner_str += f" ({stats['winner_name']})"
            
            term_reason = stats.get('termination_reason', 'normal_end')
            if game_logger:
                game_logger.log(f"[Game {i+1}/{n_games}] Score: {stats.get('p0_score', 0)}  {stats.get('p1_score', 0)}, "
                               f"Rounds: {stats.get('round_count', 0)}, Moves: {stats['move_count']}, "
                               f"Winner: {winner_str}, End: {term_reason}")
            
            logger.info(f"[Self-play] Completed game {i+1}/{n_games}")
            if i == 0:
                elapsed = time.time() - global_start
                estimated_total = elapsed * n_games
                estimated_end = time.localtime(global_start + estimated_total)
                estimated_str = time.strftime('%H:%M:%S', estimated_end)
                logger.info(f"[Self-play] Estimated completion time: {estimated_str}")
        logger.info(f"[Self-play] Completed generation of {n_games} games")
        
        # Calculate aggregate statistics
        aggregate = {
            'avg_visits': np.mean(all_stats['avg_visits']),
            'avg_entropy': np.mean(all_stats['avg_entropy']),
            'avg_reuse_rate': np.mean(all_stats['reuse_rate']),
            'avg_move_count': np.mean(all_stats['move_count'])
        }
        return all_examples, aggregate

    else:
        # MPS or CPU -> Parallel with Multiprocessing to bypass GIL
        # Each worker gets its own copy of the model
        if device.type == 'mps':
            logger.info(f"[Self-play] MPS detected ({device}), moving model to CPU for parallel execution")
            model = model.to('cpu')
            model.share_memory() # Enable shared memory for faster pickling/copying
            model.eval() # Ensure model is in eval mode for inference

        
        import torch.multiprocessing as mp
        try:
            mp.set_start_method('spawn', force=True)
        except RuntimeError:
            pass # Already set
            
        n_workers = min(8, os.cpu_count() or 1)
        logger.info(f"[Self-play] Running {n_games} games in PARALLEL via MULTIPROCESSING ({n_workers} workers)")
        
        all_examples = []
        
        # Prepare arguments for each game
        # Check opponent type
        # Check opponent type
        target_func = _run_one_game if opponent_type == 'self' else _run_one_game_vs_opponent
        
        args_with_opp = (simulations, cpuct, temperature_threshold, noise_alpha, noise_epsilon, opponent_type) if opponent_type != 'self' else (simulations, cpuct, temperature_threshold, noise_alpha, noise_epsilon)
        
        game_args = []
        for i in range(n_games):
            if opponent_type == 'self':
                game_args.append((i+1, env, model, simulations, cpuct, temperature_threshold, noise_alpha, noise_epsilon))
            else:
                 game_args.append((i+1, env, model, simulations, cpuct, temperature_threshold, noise_alpha, noise_epsilon, opponent_type))
        
        # We need a wrapper function for the pool mapping since map unpacking args is annoying
        # But we already have _run_one_game_wrapper defined below locally? No, top level is better.
        # We will use starmap.
        
        with mp.Pool(processes=n_workers) as pool:
            # chunksize=1 ensures better distribution if games vary in length
            results_iter = pool.starmap_async(target_func, game_args, chunksize=1)
            
            # Monitor progress
            total_done = 0
            while not results_iter.ready():
                time.sleep(5)
                # We can't easily get progress from starmap_async without a callback or partial results
                # But simple waiting is fine for now, we see logs from workers? 
                # Actually workers print to stdout, but might be buffered.
                # Let's just trust the process.
            
            results = results_iter.get()
            
        # Process results
        for i, (examples, stats) in enumerate(results):
            all_examples.extend(examples)
            all_stats['avg_visits'].append(stats['avg_visits'])
            all_stats['avg_entropy'].append(stats['avg_entropy'])
            all_stats['reuse_rate'].append(stats['reuse_rate'])
            all_stats['move_count'].append(stats['move_count'])
            all_stats['p0_score'].append(stats.get('p0_score', 0))
            all_stats['p1_score'].append(stats.get('p1_score', 0))
            all_stats['round_count'].append(stats.get('round_count', 0))
            
            # Log game result
            winner_str = f"P{stats.get('winner', -1)}" if stats.get('winner', -1) != -1 else "DRAW"
            if 'winner_name' in stats:
                winner_str += f" ({stats['winner_name']})"
            
            term_reason = stats.get('termination_reason', 'normal_end')
            if game_logger:
                game_logger.log(f"[Game {i+1}/{n_games}] Score: {stats.get('p0_score', 0)}  {stats.get('p1_score', 0)}, "
                               f"Rounds: {stats.get('round_count', 0)}, Moves: {stats['move_count']}, "
                               f"Winner: {winner_str}, End: {term_reason}")
        
        elapsed = time.time() - global_start
        logger.info(f"[Self-play] Completed generation of {n_games} games in {elapsed:.1f}s")
        
        # Calculate aggregate statistics
        aggregate = {
            'avg_visits': np.mean(all_stats['avg_visits']) if all_stats['avg_visits'] else 0,
            'avg_entropy': np.mean(all_stats['avg_entropy']) if all_stats['avg_entropy'] else 0,

            'avg_reuse_rate': np.mean(all_stats['reuse_rate']) if all_stats['reuse_rate'] else 0,
            'avg_move_count': np.mean(all_stats['move_count']) if all_stats['move_count'] else 0,
            'avg_p0_penalty': np.mean(all_stats['p0_floor_penalty']) if 'p0_floor_penalty' in all_stats and all_stats['p0_floor_penalty'] else 0
        }
        logger.info(f"[Self-play] Aggregate Stats: {aggregate}")
        return all_examples, aggregate