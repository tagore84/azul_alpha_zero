

import sys
import os
# Add project src folder to PYTHONPATH
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

import argparse
import torch
from torch.utils.data import DataLoader, random_split

from azul.env import AzulEnv
from net.azul_net import AzulNet
from train.self_play import generate_self_play_games
from train.dataset import AzulDataset
from train.trainer import Trainer

def main():
    parser = argparse.ArgumentParser(description="Train Azul Zero network via self-play")
    parser.add_argument('--n_games', type=int, default=50, help='Number of self-play games to generate')
    parser.add_argument('--simulations', type=int, default=100, help='MCTS simulations per move')
    parser.add_argument('--cpuct', type=float, default=1.0, help='MCTS exploration constant')
    parser.add_argument('--batch_size', type=int, default=64, help='Training batch size')
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--train_ratio', type=float, default=0.9, help='Fraction of data for training')
    parser.add_argument('--log_dir', type=str, default='logs', help='TensorBoard log directory')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints', help='Directory to save checkpoints')
    args = parser.parse_args()

    # Select device
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Initialize environment and model
    env = AzulEnv(num_players=2, factories_count=5)
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

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    trainer = Trainer(model, optimizer, device, log_dir=args.log_dir)

    # Generate self-play data
    print("Generating self-play games...")
    examples = generate_self_play_games(
        n_games=args.n_games,
        env=env,
        model=model,
        simulations=args.simulations,
        cpuct=args.cpuct
    )
    print(f"Generated {len(examples)} examples")

    # Create dataset and split
    dataset = AzulDataset(examples)
    train_size = int(len(dataset) * args.train_ratio)
    val_size = len(dataset) - train_size
    train_set, val_set = random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size)

    # Train
    trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=args.epochs,
        checkpoint_dir=args.checkpoint_dir
    )

if __name__ == "__main__":
    main()