import sys
import os
import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from azul.env import AzulEnv
from net.azul_net import AzulNet
from mcts.mcts import MCTS

def debug_mcts():
    print("Initializing AzulEnv...")
    env = AzulEnv()
    
    # Initialize Model (Random weights is fine for this test)
    obs_flat = env.encode_observation(env.reset(initial=True))
    total_obs_size = obs_flat.shape[0]
    in_channels = env.num_players * 2
    spatial_size = in_channels * 5 * 5
    factories_size = (env.N + 1) * 5
    global_size = total_obs_size - spatial_size - factories_size
    action_size = env.action_size
    
    model = AzulNet(in_channels, global_size, action_size, factories_count=env.N)
    model.eval()
    
    print("Starting MCTS Debug...")
    # Mimic validation loop
    obs = env.reset(initial=True)
    done = False
    
    # Just run one step
    print("Running MCTS step 1...")
    mcts = MCTS(env, model, simulations=25, cpuct=1.0)
    mcts.run()
    action = mcts.select_action()
    print(f"Action selected: {action}")
    
    # Check if warnings appeared
    if not mcts.root.children:
        print("FAILURE: Root children empty!")
    else:
        print(f"SUCCESS: Root children count: {len(mcts.root.children)}")

if __name__ == "__main__":
    debug_mcts()
