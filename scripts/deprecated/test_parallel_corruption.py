"""
Test if parallel self-play corrupts the model.
"""
import sys
import os
import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from azul.env import AzulEnv
from net.azul_net import AzulNet
from train.self_play import generate_self_play_games

def main():
    print("=== Parallel Self-Play Corruption Test ===")
    
    env = AzulEnv(num_players=2)
    obs_flat = env.encode_observation(env.reset())
    total_obs_size = obs_flat.shape[0]
    in_channels = env.num_players * 2
    spatial_size = in_channels * 5 * 5
    factories_size = (env.N + 1) * 5
    global_size = total_obs_size - spatial_size - factories_size
    action_size = env.action_size
    
    # Use MPS like real training
    if torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    print(f"Using device: {device}")
    
    model = AzulNet(
        in_channels=in_channels,
        global_size=global_size,
        action_size=action_size,
        factories_count=env.N
    ).to(device)
    
    # Check model before
    print("\n1. Model weights before self-play:")
    for name, param in model.named_parameters():
        if torch.isnan(param).any():
            print(f"   !! NaN: {name}")
            break
    else:
        print("   All weights clean")
    
    # Run parallel self-play like the real loop does
    print("\n2. Running parallel self-play (20 games)...")
    model.eval()
    examples, stats = generate_self_play_games(
        verbose=False,
        n_games=20,
        env=env,
        model=model,
        simulations=50,
        cpuct=1.0,
        temperature_threshold=0,
        noise_alpha=0.3,
        noise_epsilon=0.25
    )
    
    # Check model after
    print("\n3. Model weights after self-play:")
    current_device = next(model.parameters()).device
    print(f"   Model device: {current_device}")
    
    for name, param in model.named_parameters():
        if torch.isnan(param).any():
            print(f"   !! NaN: {name}")
    else:
        print("   All weights still clean")
    
    # Move back to mps and check
    if current_device.type == 'cpu':
        print("\n4. Moving model back to MPS...")
        model = model.to(device)
        for name, param in model.named_parameters():
            if torch.isnan(param).any():
                print(f"   !! NaN after move to MPS: {name}")
        else:
            print("   All weights still clean after move")
    
    # Check examples
    print(f"\n5. Generated {len(examples)} examples")
    nan_count = sum(1 for ex in examples if np.isnan(ex['obs']).any() or np.isnan(ex['pi']).any())
    print(f"   Examples with NaN: {nan_count}")
    
    print("\n=== Done ===")

if __name__ == "__main__":
    main()
