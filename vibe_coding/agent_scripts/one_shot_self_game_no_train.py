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
    print("Running one-shot self-play game no training")
    # Select device
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Initialize environment and model
    env = AzulEnv(num_players=2, factories_count=5, seed=SEED)
    #env.writer = trainer.writer
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
    #optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    #trainer = Trainer(model, optimizer, device, log_dir='logs_test_dir')
    

    # Generate self-play data
    print("Generating self-play games...")
    examples = generate_self_play_games(
        verbose=True,
        n_games=1,
        env=env,
        model=model,
        simulations=200,
        cpuct=1.0,
        show_play=True
    )
    print(f"Generated {len(examples)} examples")
    print("Saving examples to AzulDataset")
    dataset = AzulDataset(examples)
    print(dataset)

if __name__ == "__main__":
    main()