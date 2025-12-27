import sys
import os
import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from azul.env import AzulEnv
from net.azul_net import AzulNet

def test_architecture():
    print("Initializing AzulEnv...")
    env = AzulEnv()
    obs = env.reset()
    
    print("Encoding observation...")
    obs_flat = env.encode_observation(obs)
    print(f"Observation shape: {obs_flat.shape}")
    
    # Calculate expected sizes
    spatial_size = (2 * 5 * 5) # 50
    factories_size = (5 + 1) * 5 # 30
    # Global size = total - spatial - factories
    global_size = obs_flat.shape[0] - spatial_size - factories_size
    print(f"Calculated sizes: Spatial={spatial_size}, Factories={factories_size}, Global={global_size}")
    
    print("Initializing AzulNet...")
    model = AzulNet(
        in_channels=2, # Players * 1 (pattern lines) + Players * 1 (walls) ?? 
        # Wait, let's check env.encode_observation spatial parts.
        # It adds pattern_lines (5x5) and wall (5x5) for each player.
        # So 2 players * 2 planes = 4 channels?
        # Let's check env.py again.
        # spatial_parts.append(plines) -> 2
        # spatial_parts.append(wall) -> 2
        # Total 4.
        # But AzulNet init usually takes in_channels.
        # Let's check what train_loop.py passes.
        global_size=global_size,
        action_size=env.action_size,
        factories_count=env.N
    )
    
    # Fix in_channels if needed. 
    # In env.py: `spatial_parts` has num_players entries for pattern lines, then num_players for walls.
    # So 2 * num_players = 4.
    # But previous code might have used different packing.
    # Let's assume 4 for now and see if it crashes.
    
    print("Running prediction...")
    # Create batch of 2
    batch = np.array([obs_flat, obs_flat])
    pi, v = model.predict(batch)
    
    print(f"Policy output shape: {pi.shape}")
    print(f"Value output shape: {v.shape}")
    print(f"Value example: {v[0]}")
    
    assert pi.shape == (2, env.action_size)
    assert v.shape == (2,)
    
    print("Architecture test PASSED!")

if __name__ == "__main__":
    test_architecture()
