import sys
import os
import argparse
import torch
import time
from datetime import datetime

# Add project src folder to PYTHONPATH
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from azul.env import AzulEnv
from net.azul_net import AzulNet
from train.self_play import generate_self_play_games
from train.dataset import AzulDataset
from train.trainer import Trainer
from constants import SEED

def get_curriculum_params(cycle):
    """
    Returns training parameters based on the current cycle.
    Curriculum:
    - Cycles 1-5: Fast learning (Rules & Basic Tactics)
    - Cycles 6-15: Strategic learning
    - Cycles 16+: Refinement
    """
    if cycle <= 5:
        return {
            'n_games': 50,
            'simulations': 25,
            'epochs': 5,
            'lr': 1e-3,
            'cpuct': 1.0
        }
    elif cycle <= 15:
        return {
            'n_games': 100,
            'simulations': 50,
            'epochs': 10,
            'lr': 5e-4,
            'cpuct': 1.2
        }
    else:
        return {
            'n_games': 200,
            'simulations': 100,
            'epochs': 10,
            'lr': 1e-4,
            'cpuct': 1.5
        }

def main():
    parser = argparse.ArgumentParser(description="Azul Zero Training Loop")
    parser.add_argument('--total_cycles', type=int, default=20, help='Number of generation-training cycles')
    parser.add_argument('--checkpoint_dir', type=str, default='data/checkpoints', help='Directory to save models')
    parser.add_argument('--max_dataset_size', type=int, default=10000, help='Max examples in replay buffer')
    args = parser.parse_args()

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    # Setup device
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    print(f"[Loop] Using device: {device}")

    # Initialize Environment to get shapes
    env = AzulEnv(num_players=2)
    obs_flat = env.encode_observation(env.reset())
    total_obs_size = obs_flat.shape[0]
    in_channels = env.num_players * 2
    spatial_size = in_channels * 5 * 5
    global_size = total_obs_size - spatial_size
    action_size = env.action_size

    # Initialize Model
    model = AzulNet(in_channels, global_size, action_size).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    # Replay Buffer
    replay_buffer = []
    
    start_cycle = 1
    # Check for existing checkpoints to resume
    # (Simple logic: look for highest numbered model)
    # ... (Skipping complex resume logic for brevity, starting fresh or overwriting)

    for cycle in range(start_cycle, args.total_cycles + 1):
        params = get_curriculum_params(cycle)
        print(f"\n=== Cycle {cycle}/{args.total_cycles} ===")
        print(f"Params: {params}")
        
        # 1. Self-Play Generation
        print(f"[Loop] Generating {params['n_games']} games (Sims: {params['simulations']})...")
        model.eval()
        new_examples = generate_self_play_games(
            verbose=False,
            n_games=params['n_games'],
            env=env,
            model=model,
            simulations=params['simulations'],
            cpuct=params['cpuct']
        )
        
        # Add to buffer
        replay_buffer.extend(new_examples)
        if len(replay_buffer) > args.max_dataset_size:
            replay_buffer = replay_buffer[-args.max_dataset_size:]
        print(f"[Loop] Buffer size: {len(replay_buffer)}")
        
        # 2. Training
        print(f"[Loop] Training for {params['epochs']} epochs...")
        dataset = AzulDataset(replay_buffer, augment_factories=True)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True)
        
        # Update Learning Rate
        for param_group in optimizer.param_groups:
            param_group['lr'] = params['lr']
            
        trainer = Trainer(model, optimizer, device, log_dir=f'logs/cycle_{cycle}')
        trainer.fit(dataloader, epochs=params['epochs'])
        
        # 3. Checkpoint
        ckpt_path = os.path.join(args.checkpoint_dir, f"model_cycle_{cycle}.pt")
        torch.save({
            'cycle': cycle,
            'model_state': model.state_dict(),
            'optimizer_state': optimizer.state_dict(),
            'params': params
        }, ckpt_path)
        print(f"[Loop] Saved checkpoint to {ckpt_path}")
        
        # Save latest as 'best.pt' for easy access
        torch.save({'model_state': model.state_dict()}, os.path.join(args.checkpoint_dir, "best.pt"))

if __name__ == "__main__":
    main()
