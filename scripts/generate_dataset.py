

import os
import argparse
import torch
from datetime import datetime

from constants import SEED
from azul.env import AzulEnv
from net.azul_net import AzulNet
from train.self_play import generate_self_play_games

MACHINE_ID = os.environ.get("AZUL_MACHINE_ID", "default")

def main():
    parser = argparse.ArgumentParser(description="Generate self-play dataset")
    parser.add_argument('--n_games', type=int, default=100, help='Number of self-play games to generate')
    parser.add_argument('--simulations', type=int, default=200, help='MCTS simulations per move')
    parser.add_argument('--cpuct', type=float, default=1.0, help='MCTS exploration constant')
    parser.add_argument('--checkpoint_dir', type=str, default='data/checkpoint_dir', help='Directory to store replay buffer')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose output')
    args = parser.parse_args()

    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    print(f"Using device: {device}")

    env = AzulEnv(num_players=2, factories_count=5, seed=SEED)
    sample_obs = env.reset()
    obs_flat = env.encode_observation(sample_obs)
    total_obs_size = obs_flat.shape[0]
    in_channels = total_obs_size // (5 * 5)
    spatial_size = in_channels * 5 * 5
    global_size = total_obs_size - spatial_size
    print(f"Obs total size: {total_obs_size}, spatial_size: {spatial_size}, global_size: {global_size}, in_channels: {in_channels}")
    action_size = env.action_size

    model = AzulNet(
        in_channels=in_channels,
        spatial_size=spatial_size,
        global_size=global_size,
        action_size=action_size
    ).to(device)
    model.eval()

    print(f"[{datetime.now().strftime('%H:%M:%S')}] Generating self-play games...")
    examples = generate_self_play_games(
        verbose=args.verbose,
        n_games=args.n_games,
        env=env,
        model=model,
        simulations=args.simulations,
        cpuct=args.cpuct
    )
    print(f"Generated {len(examples)} examples")

    replay_buffer_path = os.path.join(args.checkpoint_dir, f'replay_buffer_{MACHINE_ID}.pt')
    torch.save({'examples': examples}, replay_buffer_path)
    print(f"Saved replay buffer to {replay_buffer_path}")

if __name__ == "__main__":
    main()