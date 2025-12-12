
import sys
import os
import torch
import time
import logging

# Add project src folder to PYTHONPATH
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from azul.env import AzulEnv
from net.azul_net import AzulNet
from train.self_play import play_game_vs_opponent

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    print("=== DEBUG SINGLE GAME START ===")
    
    # 1. Initialize Env
    env = AzulEnv()
    
    # 2. Initialize Model (Random Weights)
    obs = env.reset()
    obs_flat = env.encode_observation(obs)
    total_obs_size = obs_flat.shape[0]
    in_channels = 4
    spatial_size = 100
    factories_size = (env.N + 1) * 5
    global_size = total_obs_size - spatial_size - factories_size
    
    device = torch.device('cpu')
    model = AzulNet(
        in_channels=in_channels,
        global_size=global_size,
        action_size=env.action_size,
        factories_count=env.N
    ).to(device)
    model.eval()
    
    # 3. Play Game
    print("Starting game execution...")
    start = time.time()
    
    # Use small sims for speed
    examples, stats = play_game_vs_opponent(
        env, 
        model, 
        simulations=10, 
        cpuct=1.0, 
        opponent_type='random'
    )
    
    end = time.time()
    print(f"=== DEBUG SINGLE GAME END ===")
    print(f"Time taken: {end - start:.2f}s")
    print(f"Game Stats: {stats}")
    print(f"Examples count: {len(examples)}")
    
    if len(examples) > 0:
        print("Example 0 V:", examples[0]['v'])

if __name__ == "__main__":
    main()
