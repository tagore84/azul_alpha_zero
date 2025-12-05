"""
Debug script to reproduce the exact training step.
"""
import sys
import os
import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from azul.env import AzulEnv
from net.azul_net import AzulNet
from train.self_play import play_game
from train.dataset import AzulDataset

def main():
    print("=== Training Step Debug ===")
    
    # Create fresh model
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
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    # Generate 1 game of data
    print("\n1. Generating 1 self-play game...")
    examples, stats = play_game(
        env.clone(), 
        model.to('cpu'),  # Self-play uses CPU
        simulations=50,
        cpuct=1.0,
        temperature_threshold=0,
        noise_alpha=0.3,
        noise_epsilon=0.25
    )
    model = model.to(device)
    print(f"   Generated {len(examples)} examples")
    
    # Create dataset and dataloader
    print("\n2. Creating dataset...")
    dataset = AzulDataset(examples, augment_factories=True)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True)
    
    # Training step
    print("\n3. Running training epoch...")
    model.train()
    
    for batch_idx, batch in enumerate(dataloader):
        obs_spatial = batch['spatial'].to(device)
        obs_factories = batch['factories'].to(device)
        obs_global = batch['global'].to(device)
        target_pi = batch['pi'].to(device)
        target_v = batch['v'].to(device).float()
        
        print(f"\n   Batch {batch_idx}:")
        print(f"   - spatial shape: {obs_spatial.shape}, range: [{obs_spatial.min():.2f}, {obs_spatial.max():.2f}]")
        print(f"   - factories shape: {obs_factories.shape}, range: [{obs_factories.min():.2f}, {obs_factories.max():.2f}]")
        print(f"   - global shape: {obs_global.shape}, range: [{obs_global.min():.2f}, {obs_global.max():.2f}]")
        print(f"   - target_pi sum range: [{target_pi.sum(dim=1).min():.4f}, {target_pi.sum(dim=1).max():.4f}]")
        print(f"   - target_v range: [{target_v.min():.2f}, {target_v.max():.2f}]")
        
        # Check for NaN in inputs
        if torch.isnan(obs_spatial).any():
            print("   !! NaN in obs_spatial")
        if torch.isnan(obs_factories).any():
            print("   !! NaN in obs_factories")
        if torch.isnan(obs_global).any():
            print("   !! NaN in obs_global")
        if torch.isnan(target_pi).any():
            print("   !! NaN in target_pi")
        if torch.isnan(target_v).any():
            print("   !! NaN in target_v")
        
        # Forward pass
        pi_logits, value = model(obs_spatial, obs_global, obs_factories)
        
        print(f"   - pi_logits range: [{pi_logits.min():.2f}, {pi_logits.max():.2f}]")
        print(f"   - value range: [{value.min():.2f}, {value.max():.2f}]")
        
        if torch.isnan(pi_logits).any():
            print("   !! NaN in pi_logits OUTPUT")
        if torch.isnan(value).any():
            print("   !! NaN in value OUTPUT")
        
        # Compute losses
        log_pi = torch.nn.functional.log_softmax(pi_logits, dim=1)
        
        if torch.isnan(log_pi).any():
            print("   !! NaN in log_softmax")
            # Check for -inf in log_pi
            if torch.isinf(log_pi).any():
                print("   !! Inf in log_softmax (expected for masked actions)")
        
        l_pi = -(target_pi * log_pi).sum(dim=1).mean()
        l_v = torch.nn.functional.mse_loss(value, target_v)
        loss = l_pi + l_v
        
        print(f"   - loss: {loss.item():.4f} (pi: {l_pi.item():.4f}, v: {l_v.item():.4f})")
        
        if torch.isnan(loss):
            print("   !! NaN LOSS DETECTED")
            # Check what went wrong
            print(f"   Checking cross-entropy term: (target_pi * log_pi)")
            cross_ent = (target_pi * log_pi)
            if torch.isnan(cross_ent).any():
                print("   !! NaN in cross-entropy product")
                # This happens when target_pi > 0 and log_pi = -inf
                # i.e., network predicts 0 probability for an action that MCTS visited
                bad_idx = torch.isnan(cross_ent).any(dim=1)
                print(f"   Bad samples: {bad_idx.sum().item()}")
            break
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        
        # Check gradients
        has_nan_grad = False
        for name, param in model.named_parameters():
            if param.grad is not None and torch.isnan(param.grad).any():
                print(f"   !! NaN gradient in {name}")
                has_nan_grad = True
                
        if has_nan_grad:
            print("   !! NaN gradients detected, stopping")
            break
            
        optimizer.step()
        
        # Check weights after step
        for name, param in model.named_parameters():
            if torch.isnan(param).any():
                print(f"   !! NaN in weights after step: {name}")
                break
        else:
            print("   Weights OK after step")
            
        if batch_idx >= 2:  # Only check first few batches
            break
    
    print("\n=== Done ===")

if __name__ == "__main__":
    main()
