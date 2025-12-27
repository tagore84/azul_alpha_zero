import torch
import os
import sys

# Add project src folder to PYTHONPATH
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from azul.env import AzulEnv
from net.azul_net import AzulNet

def main():
    print("Initializing dummy model...")
    device = 'cpu'
    
    # Initialize env to get dimensions
    env = AzulEnv()
    obs_flat = env.encode_observation(env.reset(initial=True))
    total_obs_size = obs_flat.shape[0]
    spatial_size = env.num_players * 2 * 5 * 5
    factories_size = (env.N + 1) * 5
    global_size = total_obs_size - spatial_size - factories_size
    action_size = env.action_size
    
    # Initialize model
    model = AzulNet(
        in_channels=env.num_players * 2,
        global_size=global_size,
        action_size=action_size,
        factories_count=env.N
    ).to(device)
    
    # Save model
    save_dir = os.path.join(os.path.dirname(__file__), '../data/checkpoints_v5')
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, 'best.pt')
    
    checkpoint = {
        'model_state': model.state_dict(),
        'optimizer_state': {},
        'cycle': 0
    }
    
    torch.save(checkpoint, save_path)
    print(f"Dummy model saved to {save_path}")

if __name__ == "__main__":
    main()
