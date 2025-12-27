
import sys
import os
import torch
import numpy as np

# Add project src folder to PYTHONPATH
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from azul.env import AzulEnv
from net.azul_net import AzulNet
from train.self_play import play_game

def verify():
    print("Verifying Self-Play Fix...")
    
    # Init Env and Model
    env = AzulEnv()
    
    # Load Cycle 2 model (the one that was stalling)
    checkpoint_path = "data/checkpoints_v5/model_cycle_2.pt"
    if not os.path.exists(checkpoint_path):
        print("Checkpoint not found, cannot verify with trained model behavior.")
        return

    device = torch.device('cpu') # Use CPU for verification
    
    # Init Model structure
    obs_flat = env.encode_observation(env.reset())
    total_obs_size = obs_flat.shape[0]
    in_channels = env.num_players * 2 * 5 # 20
    spatial_size = in_channels * 5 * 5 
    factories_size = (env.N + 1) * 5 
    global_size = total_obs_size - spatial_size - factories_size
    action_size = env.action_size
    
    model = AzulNet(
        in_channels=in_channels, 
        global_size=global_size, 
        action_size=action_size, 
        factories_count=env.N
    ).to(device)
    
    print(f"Loading checkpoint {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state'])
    model.eval()
    
    # Run 1 Game with the new Logic (single_player_mode=True is now hardcoded in self_play.py)
    print("Running 1 Self-Play game...")
    # Using 100 sims to be quick but relatively smart
    examples, stats = play_game(env, model, simulations=100, cpuct=1.0)
    
    print("\n=== Verification Result ===")
    print(f"Rounds: {stats['round_count']}")
    print(f"Score: {stats['p0_score']} vs {stats['p1_score']}")
    print(f"Winner: {stats['winner']}")
    print(f"Moves: {stats['move_count']}")
    
    if stats['round_count'] < 15:
        print("\nSUCCESS: Game finished in reasonable rounds!")
    else:
        print("\nFAILURE: Game still took too many rounds!")

if __name__ == "__main__":
    verify()
