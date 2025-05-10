

import numpy as np
import torch
from typing import Any, List, Dict
from azul.env import AzulEnv
from mcts.mcts import MCTS

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
        mcts.advance(action)

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
    n_games: int,
    env: AzulEnv,
    model: Any,
    simulations: int = 100,
    cpuct: float = 1.0
) -> List[Dict[str, Any]]:
    """
    Generate multiple self-play games.
    Returns a flat list of training examples.
    """
    all_examples = []
    for _ in range(n_games):
        # Clone the environment for a fresh game
        env_clone = env.__class__(num_players=env.num_players, factories_count=env.N)
        env_clone.__dict__ = env.__dict__.copy()
        examples = play_game(env_clone, model, simulations, cpuct)
        all_examples.extend(examples)
    return all_examples