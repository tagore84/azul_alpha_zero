"""
Debug script to check if self-play generates NaN data.
Run this BEFORE starting training to verify data is clean.
"""
import sys
import os
import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from azul.env import AzulEnv
from net.azul_net import AzulNet
from train.self_play import play_game

def check_examples_for_nan(examples):
    """Check if examples contain NaN or Inf values."""
    issues = []
    for i, ex in enumerate(examples):
        obs = ex['obs']
        pi = ex['pi']
        v = ex['v']
        
        if np.isnan(obs).any():
            issues.append(f"Example {i}: NaN in obs")
        if np.isinf(obs).any():
            issues.append(f"Example {i}: Inf in obs")
        if np.isnan(pi).any():
            issues.append(f"Example {i}: NaN in pi")
        if np.isinf(pi).any():
            issues.append(f"Example {i}: Inf in pi")
        if np.isnan(v):
            issues.append(f"Example {i}: NaN in v")
        if np.isinf(v):
            issues.append(f"Example {i}: Inf in v")
            
        # Check for extreme values
        if np.abs(obs).max() > 1e6:
            issues.append(f"Example {i}: Extreme value in obs: {np.abs(obs).max()}")
        if np.abs(pi).max() > 1e6:
            issues.append(f"Example {i}: Extreme value in pi: {np.abs(pi).max()}")
            
    return issues

def main():
    print("=== Self-Play Data Debug ===")
    
    # Create fresh model
    env = AzulEnv(num_players=2)
    obs_flat = env.encode_observation(env.reset())
    total_obs_size = obs_flat.shape[0]
    in_channels = env.num_players * 2
    spatial_size = in_channels * 5 * 5
    factories_size = (env.N + 1) * 5
    global_size = total_obs_size - spatial_size - factories_size
    action_size = env.action_size
    
    print(f"Obs size: {total_obs_size}, Action size: {action_size}")
    
    # Use CPU manually for debugging
    device = torch.device('cpu')
    
    model = AzulNet(
        in_channels=in_channels,
        global_size=global_size,
        action_size=action_size,
        factories_count=env.N
    ).to(device)
    
    # Check initial model for NaN
    print("\n1. Checking fresh model weights...")
    for name, param in model.named_parameters():
        if torch.isnan(param).any():
            print(f"!! NaN in fresh model: {name}")
            return
    print("Fresh model is clean.")
    
    # Generate a few games
    print("\n2. Generating 3 self-play games...")
    all_examples = []
    for i in range(3):
        print(f"  Game {i+1}...")
        examples, stats = play_game(
            env.clone(), 
            model, 
            simulations=50,  # Low for speed
            cpuct=1.0,
            temperature_threshold=0,
            noise_alpha=0.3,
            noise_epsilon=0.25
        )
        all_examples.extend(examples)
        print(f"    Generated {len(examples)} examples")
    
    print(f"\n3. Checking {len(all_examples)} examples for NaN/Inf...")
    issues = check_examples_for_nan(all_examples)
    
    if issues:
        print("\n!! ISSUES FOUND:")
        for issue in issues[:20]:  # Show first 20
            print(f"   {issue}")
        if len(issues) > 20:
            print(f"   ... and {len(issues) - 20} more")
    else:
        print("All examples are clean (no NaN/Inf).")
        
    # Show sample data ranges
    print("\n4. Data ranges:")
    obs_vals = np.concatenate([ex['obs'] for ex in all_examples])
    pi_vals = np.concatenate([ex['pi'] for ex in all_examples])
    v_vals = np.array([ex['v'] for ex in all_examples])
    
    print(f"   obs: min={obs_vals.min():.4f}, max={obs_vals.max():.4f}, mean={obs_vals.mean():.4f}")
    print(f"   pi:  min={pi_vals.min():.4f}, max={pi_vals.max():.4f}, sum_range=[{pi_vals.reshape(-1, action_size).sum(axis=1).min():.4f}, {pi_vals.reshape(-1, action_size).sum(axis=1).max():.4f}]")
    print(f"   v:   min={v_vals.min():.4f}, max={v_vals.max():.4f}")

if __name__ == "__main__":
    main()
