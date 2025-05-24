# scripts/one_shot_self_game.py

import sys
import os
# Add project src folder to PYTHONPATH
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from constants import SEED
import argparse
import torch
from torch.utils.data import DataLoader, random_split

import copy
from mcts.mcts import MCTS
from net.azul_net import AzulNet
from azul.env import AzulEnv
from net.azul_net import AzulNet
from train.self_play import generate_self_play_games
from train.dataset import AzulDataset
from train.trainer import Trainer


def main():
    parser = argparse.ArgumentParser(description="Just one self-play")
    parser.add_argument('--verbose', action='store_true', help='Enable verbose logging')
    parser.add_argument('--n_games', type=int, default=50, help='Number of self-play games to generate')
    parser.add_argument('--simulations', type=int, default=100, help='MCTS simulations per move')
    parser.add_argument('--cpuct', type=float, default=1.0, help='MCTS exploration constant')
    parser.add_argument('--batch_size', type=int, default=64, help='Training batch size')
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--train_ratio', type=float, default=0.9, help='Fraction of data for training')
    parser.add_argument('--log_dir', type=str, default='logs', help='TensorBoard log directory')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints', help='Directory to save checkpoints')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to a model checkpoint to resume training from')
    parser.add_argument('--eval_interval', type=int, default=5,
                        help='Number of epochs between self-play evaluations')
    parser.add_argument('--eval_games',    type=int, default=100,
                        help='Number of games to play in each evaluation')
    args = parser.parse_args()

    prev_checkpoint = args.resume if args.resume else None

    # Select device
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Initialize environment and model
    env = AzulEnv(num_players=2, factories_count=5, seed=SEED)
    # Dynamically compute observation sizes from a sample reset
    sample_obs = env.reset()
    obs_flat = env.encode_observation(sample_obs)
    total_obs_size = obs_flat.shape[0]
    # Determine number of spatial channels (must divide by 5*5)
    in_channels = total_obs_size // (5 * 5)
    spatial_size = in_channels * 5 * 5
    global_size = total_obs_size - spatial_size
    print(f"Obs total size: {total_obs_size}, spatial_size: {spatial_size}, global_size: {global_size}, in_channels: {in_channels}")
    action_size = env.action_size

    model = AzulNet(
        in_channels=in_channels,
        global_size=global_size,
        action_size=action_size
    )
    model = model.to(device)


    # Generate self-play data
    print("Generating self-play games...")
    examples = generate_self_play_games(
        n_games=1,
        env=env,
        model=model,
        simulations=1,
        cpuct=args.cpuct,
        verbose=args.verbose
    )
    print(f"Generated {len(examples)} examples")
    print("Saving examples to AzulDataset")
    dataset = AzulDataset(examples)
    print(dataset)

if __name__ == "__main__":
    main()