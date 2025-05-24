
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from players.maximilian_times import MaximilianTimes
from players.lillo_expertillo import LilloExpertillo
from players.random_player import RandomPlayer
import argparse
from players.deep_mcts_player import DeepMCTSPlayer 
from players.expert_player import ExpertPlayer
from players.heuristic_player import HeuristicPlayer
from azul.env import AzulEnv


def render_obs(obs: dict, mode='human'):
    """
    Render the game state from an observation dict (as returned by _get_obs).
    """
    print(f"Player to move: {obs['current_player']}")
    for idx, p in enumerate(obs['players']):
        print(f"== Player {idx} ==")
        print("Score:", p['score'])
        # Wall and pattern lines
        print("Wall:\n", p['wall'])
        print("Pattern lines:")
        for line in p['pattern_lines']:
            print(" ", line)
        print("Floor line:", p['floor_line'])
    # Factories and center
    print("Factories:\n", obs['factories'])
    print("Center:", obs['center'], "First token present:", obs['first_player_token'])


if __name__ == "__main__":
    p1 = DeepMCTSPlayer("data/checkpoint_dir/checkpoint_latest_mac.pt", device="cpu", mcts_iters=50, cpuct=0.5)
    p2 = LilloExpertillo()
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
    print(env.get_final_scores())


