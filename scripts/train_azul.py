import sys
import os
# Add project src folder to PYTHONPATH
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

import argparse
import torch
from torch.utils.data import DataLoader, random_split

import copy
from mcts.mcts import MCTS
from net.azul_net import AzulNet
from azul.env import AzulEnv

from azul.env import AzulEnv
from net.azul_net import AzulNet
from train.self_play import generate_self_play_games
from train.dataset import AzulDataset
from train.trainer import Trainer

def evaluate_against_previous(current_model, previous_model, env_args, simulations, cpuct, n_games):
    """
    Play n_games between current_model (player 0) and previous_model (player 1).
    Returns wins_current, wins_previous.
    """
    wins_current = 0
    wins_prev    = 0
    for _ in range(n_games):
        env = AzulEnv(**env_args)
        obs = env.reset()
        done = False
        while not done:
            current = env.current_player - 1  # 0 or 1
            model = current_model if current == 0 else previous_model
            mcts = MCTS(
                env.__class__(num_players=env.num_players, factories_count=env.N),
                model, simulations=simulations, cpuct=cpuct
            )
            mcts.root.env.__dict__ = copy.copy(env.__dict__)
            mcts.run()
            action = mcts.select_action()
            obs, reward, done, info = env.step(action)
        total_rewards = info.get('total_rewards', [0,0])
        if total_rewards[0] > total_rewards[1]:
            wins_current += 1
        else:
            wins_prev += 1
    return wins_current, wins_prev

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
    for epoch in range(1, args.epochs + 1):
        trainer.fit(
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=1,
            checkpoint_dir=args.checkpoint_dir
        )
        checkpoint_path = os.path.join(args.checkpoint_dir, f"model_epoch_{epoch}.pt")
        torch.save({'model_state': model.state_dict()}, checkpoint_path)
        print(f"Saved checkpoint: {checkpoint_path}")

        # Periodic evaluation against previous checkpoint
        if prev_checkpoint and (epoch % args.eval_interval == 0):
            prev_model = AzulNet(in_channels, global_size, action_size)
            prev_model.load_state_dict(torch.load(prev_checkpoint, map_location=device)['model_state'])
            prev_model.to(device)
            current_model = copy.deepcopy(model)
            current_model.eval()
            prev_model.eval()
            wins_current, wins_prev = evaluate_against_previous(
                current_model, prev_model,
                {'num_players': env.num_players, 'factories_count': env.N},
                simulations=args.simulations, cpuct=args.cpuct, n_games=args.eval_games
            )
            print(f"Eval at epoch {epoch}: current wins {wins_current}, previous wins {wins_prev}")
            trainer.writer.add_scalar('eval/current_wins', wins_current, epoch)
            trainer.writer.add_scalar('eval/previous_wins', wins_prev, epoch)
        prev_checkpoint = checkpoint_path

if __name__ == "__main__":
    main()