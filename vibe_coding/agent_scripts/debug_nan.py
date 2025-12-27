import sys
import os
import torch
import numpy as np
from torch.utils.data import DataLoader

# Add project src folder to PYTHONPATH
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from azul.env import AzulEnv
from net.azul_net import AzulNet
from train.dataset import AzulDataset
from train.trainer import Trainer

def main():
    print("[Debug] Starting NaN Reproduction...")
    
    # 1. Setup Device
    device = torch.device('cpu') # Use CPU for anomaly detection (better trace)
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    
    print(f"[Debug] Device: {device}")

    # 2. Initialize Env for shapes
    env = AzulEnv(num_players=2)
    obs = env.reset()
    obs_flat = env.encode_observation(obs)
    
    total_obs_size = obs_flat.shape[0]
    in_channels = env.num_players * 2 # 4
    spatial_size = in_channels * 5 * 5 # 100
    factories_size = (env.N + 1) * 5 # 30
    global_size = total_obs_size - spatial_size - factories_size
    action_size = env.action_size
    
    print(f"[Debug] Shapes: Spatial={spatial_size}, Factories={factories_size}, Global={global_size}")

    # 3. Load Model
    model = AzulNet(
        in_channels=in_channels,
        global_size=global_size,
        action_size=action_size,
        factories_count=env.N
    ).to(device)
    
    ckpt_path = "data/checkpoints_v5/model_cycle_9.pt"
    if os.path.exists(ckpt_path):
        print(f"[Debug] Loading checkpoint {ckpt_path}...")
        checkpoint = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(checkpoint['model_state'])
    else:
        print("[Debug] Checkpoint not found! Using random initialization (might not reproduce).")

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3) # High-ish LR

    # 4. Generate Synthetic Data
    # We want to mimic the "Speedrun" / Identical batch issue?
    # Or just random valid data.
    
    replay_buffer = []
    
    print("[Debug] Generating synthetic data...")
    for _ in range(100):
        # Create a valid observation
        env.reset()
        # Random moves to get interesting state
        for _ in range(random.randint(0, 20)):
            valid = env.get_valid_actions()
            if not valid: break
            env.step(random.choice(valid))
            
        obs_flat = env.encode_observation(env._get_obs())
        
        # Policy target: one-hot or random?
        valid = env.get_valid_actions()
        pi = np.zeros(action_size, dtype=np.float32)
        if valid:
            for v in valid:
                idx = env.action_to_index(v)
                pi[idx] = 1.0 / len(valid) # Uniform over valid
        else:
            pi[0] = 1.0 # Fallback
            
        # Mask
        mask = np.zeros(action_size, dtype=np.float32)
        if valid:
            for v in valid:
                idx = env.action_to_index(v)
                mask[idx] = 1.0
        
        # Value target: random in [-1, 1]
        v = random.uniform(-1.0, 1.0)
        
        replay_buffer.append({
            'obs': obs_flat,
            'pi': pi,
            'v': np.float32(v),
            'mask': mask
        })

    dataset = AzulDataset(replay_buffer, augment_factories=False)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
    
    # 5. Run Training with Anomaly Detection
    print("[Debug] Enabling Anomaly Detection...")
    torch.autograd.set_detect_anomaly(True)
    
    trainer = Trainer(model, optimizer, device)
    
    # Run one epoch manually to catch error
    model.train()
    
    try:
        for batch_idx, batch in enumerate(dataloader):
            print(f"[Debug] Batch {batch_idx}")
            
            obs_spatial   = batch['spatial'].to(device)
            obs_factories = batch['factories'].to(device)
            obs_global    = batch['global'].to(device)
            target_pi     = batch['pi'].to(device)
            target_v      = batch['v'].to(device).float()
            action_mask   = batch['mask'].to(device)

            # Forward
            pi_logits, value = model(obs_spatial, obs_global, obs_factories, action_mask=action_mask)
            
            log_pi = torch.nn.functional.log_softmax(pi_logits, dim=1)
            l_pi = -(target_pi * log_pi).sum(dim=1).mean()
            l_v  = torch.nn.functional.mse_loss(value, target_v)
            loss = l_pi + l_v
            
            print(f"[Debug] Loss: {loss.item()}")
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            print("[Debug] Step executed successfully.")
            
    except RuntimeError as e:
        print("\n\n################################################")
        print("CAUGHT RUNTIME ERROR (Potential NaN source):")
        print(e)
        print("################################################\n")
        raise e

import random
if __name__ == "__main__":
    main()
