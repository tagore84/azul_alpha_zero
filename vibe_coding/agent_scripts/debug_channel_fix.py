
import sys
import os
import torch
from torch.utils.data import DataLoader

# Add project src folder to PYTHONPATH
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from azul.env import AzulEnv
from net.azul_net import AzulNet
from train.dataset import AzulDataset

def test_fix():
    print("Initializing Env...")
    env = AzulEnv(num_players=2)
    obs_flat = env.encode_observation(env.reset())
    
    print(f"Observation Shape: {obs_flat.shape}")
    
    # Mock data
    example = {
        'obs': obs_flat,
        'pi':  torch.zeros(env.action_size).numpy(),
        'v':   0.0,
        'mask': torch.ones(env.action_size).numpy()
    }
    
    # Create Dataset
    print("Creating Dataset...")
    dataset = AzulDataset([example])
    dataloader = DataLoader(dataset, batch_size=1)
    
    # Get Batch
    batch = next(iter(dataloader))
    
    spatial = batch['spatial']
    print(f"Spatial Batch Shape: {spatial.shape}")
    
    expected_channels = 20
    if spatial.shape[1] != expected_channels:
        print(f"ERROR: Expected {expected_channels} channels, got {spatial.shape[1]}")
        sys.exit(1)
        
    print("Initializing Model...")
    model = AzulNet(
        in_channels=expected_channels,
        global_size=batch['global'].shape[1],
        action_size=env.action_size,
        factories_count=env.N
    )
    
    print("Running Forward Pass...")
    try:
        model(batch['spatial'], batch['global'], batch['factories'], batch['mask'])
        print("SUCCESS: Forward pass completed without error.")
    except Exception as e:
        print(f"FAILURE: {e}")
        sys.exit(1)

if __name__ == "__main__":
    test_fix()
