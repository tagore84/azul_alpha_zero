import sys
import os
from datetime import datetime
# Add project src folder to PYTHONPATH
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from constants import SEED
import argparse
import torch
from torch.utils.data import DataLoader, random_split

import copy

from net.azul_net import AzulNet, evaluate_against_previous
from azul.env import AzulEnv
from train.self_play import generate_self_play_games
from train.dataset import AzulDataset
from train.trainer import Trainer
import random
    


def main():
    parser = argparse.ArgumentParser(description="Train Azul Zero network via self-play")
    parser.add_argument('--verbose', type=bool, default=False, help='Logging verbosity')
    parser.add_argument('--n_games', type=int, default=100, help='Number of self-play games to generate')
    parser.add_argument('--simulations', type=int, default=200, help='MCTS simulations per move')
    parser.add_argument('--cpuct', type=float, default=1.0, help='MCTS exploration constant')
    parser.add_argument('--checkpoint_dir', type=str, default='data/checkpoint_dir', help='Directory to save checkpoints')
    parser.add_argument('--base_model', type=str, default=None,
                        help='Path to a model checkpoint to resume training from')
    parser.add_argument('--resume_training', action='store_true', help='Append to existing last_dataset.pt if it exists')
    args = parser.parse_args()

    if SEED is not None:
        print(f"[generate-dataset] SEED: {SEED}")

    base_model = None
    if args.base_model:
        base_model = args.base_model
        
    # Select device
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    print(f"[generate-dataset] Using device: {device}")

    # Initialize environment and model
    env = AzulEnv(num_players=2, factories_count=5, seed=SEED)
    # Dynamically compute observation sizes from a sample reset
    sample_obs = env.reset()
    obs_flat = env.encode_observation(sample_obs)
    total_obs_size = obs_flat.shape[0]
    # Determine number of spatial channels
    in_channels = env.num_players * 2
    spatial_size = in_channels * 5 * 5
    global_size = total_obs_size - spatial_size
    print(f"[generate-dataset] Obs total size: {total_obs_size}, spatial_size: {spatial_size}, global_size: {global_size}, in_channels: {in_channels}")
    action_size = env.action_size

    model = AzulNet(
        in_channels=in_channels,
        global_size=global_size,
        action_size=action_size
    )
    model = model.to(device)
    if base_model:
        checkpoint = torch.load(base_model, map_location=device)
        state_dict = checkpoint.get('model_state',
                       checkpoint.get('state_dict', checkpoint))
        model.load_state_dict(state_dict)
    

    # Generate self-play data
    print(f"[generate-dataset] [{datetime.now().strftime('%H:%M:%S')}] Generating self-play games...")
    new_examples = generate_self_play_games(
        verbose=args.verbose,
        n_games=args.n_games,
        env=env,
        model=model,
        simulations=args.simulations,
        cpuct=args.cpuct
    )
    print(f"[generate-dataset] Generated {len(new_examples)} examples")
    last_dataset_path = os.path.join(args.checkpoint_dir, 'last_dataset.pt')
    if args.resume_training and os.path.exists(last_dataset_path):
        #previous_data = torch.load(last_dataset_path)
        previous_data = torch.load(last_dataset_path, weights_only=False)
        previous_examples = previous_data.get('examples', [])
        combined_examples = previous_examples + new_examples
        print(f"[generate-dataset] Loading existing dataset {len(previous_examples)} examples, total {len(new_examples)+len(previous_examples)} examples")
    else:
        combined_examples = new_examples

    torch.save({'examples': combined_examples}, last_dataset_path)
    
if __name__ == "__main__":
    main()