import sys
import os
from datetime import datetime
MACHINE_ID = os.environ.get("AZUL_MACHINE_ID", "default")
# Add project src folder to PYTHONPATH
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from constants import SEED
import argparse
import torch
import copy

from net.azul_net import AzulNet, evaluate_against_previous
from train.dataset import AzulDataset
from train.trainer import Trainer


def main():
    parser = argparse.ArgumentParser(description="Train Azul Zero network via self-play")
    parser.add_argument('--verbose', type=bool, default=False, help='Logging verbosity')
    parser.add_argument('--batch_size', type=int, default=64, help='Training batch size')
    parser.add_argument('--epochs', type=int, default=20, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--train_ratio', type=float, default=0.9, help='Fraction of data for training')
    parser.add_argument('--log_dir', type=str, default='logs', help='TensorBoard log directory')
    parser.add_argument('--checkpoint_dir', type=str, default='data/checkpoint_dir', help='Directory to save checkpoints')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to a model checkpoint to resume training from')
    parser.add_argument('--eval_interval', type=int, default=10,
                        help='Number of epochs between self-play evaluations')
    parser.add_argument('--eval_games',    type=int, default=20,
                        help='Number of games to play in each evaluation')
    parser.add_argument('--buffer', type=str, default=None, help='Path to the replay buffer to load')
    args = parser.parse_args()
    
    # Select device
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    print(f"Using device: {device}")

    prev_checkpoint = None
    best_checkpoint = os.path.join(args.checkpoint_dir, 'checkpoint_best.pt')
    if args.resume:
        checkpoint_path = args.resume
        print(f"Resuming from checkpoint: {checkpoint_path}")
        prev_checkpoint = checkpoint_path
    elif os.path.exists(best_checkpoint):
        print(f"Auto-loading best checkpoint from {best_checkpoint}")
        prev_checkpoint = best_checkpoint
    else:
        default_latest = os.path.join(args.checkpoint_dir, f'checkpoint_latest_{MACHINE_ID}.pt')
        if os.path.exists(default_latest):
            print(f"Auto-loading latest checkpoint from {default_latest}")
            prev_checkpoint = default_latest

    if not args.buffer or not os.path.exists(args.buffer):
        raise ValueError(f"Replay buffer not found: {args.buffer}")
    with open(args.buffer, 'rb') as f:
        saved = torch.load(args.buffer, map_location=device, weights_only=False)
    examples = saved['examples']
    print(f"Loaded examples type: {type(examples)}, length: {len(examples)}")
    if len(examples) > 0:
        print(f"First example: {examples[0]}")

    dataset = AzulDataset(examples.copy())
    train_size = int(len(dataset) * args.train_ratio)
    val_size = len(dataset) - train_size
    from torch.utils.data import DataLoader, random_split
    train_set, val_set = random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size)

    # Initialize environment sizes from saved examples
    # We assume the environment sizes are saved in the checkpoint or can be inferred
    # Here we infer from example observation shape
    print(f"Loaded examples type: {type(examples)}, length: {len(examples)}")
    obs_example = examples[0]['obs']
    total_obs_size = obs_example.shape[0]
    in_channels = total_obs_size // (5 * 5)
    spatial_size = in_channels * 5 * 5
    global_size = total_obs_size - spatial_size
    # Action size must be saved or known; here we load from a checkpoint if possible
    if prev_checkpoint:
        checkpoint = torch.load(prev_checkpoint, map_location=device)
        action_size = checkpoint.get('action_size', None)
    else:
        action_size = None
    if action_size is None:
        # Fallback: infer action_size from examples (assuming action is the index of the highest probability in 'pi')
        action_size = max([ex['pi'].argmax().item() for ex in examples]) + 1

    model = AzulNet(
        in_channels=in_channels,
        global_size=global_size,
        action_size=action_size
    )
    model = model.to(device)
    if prev_checkpoint:
        checkpoint = torch.load(prev_checkpoint, map_location=device)
        # Support checkpoints with different key names
        state_dict = checkpoint.get('model_state',
                       checkpoint.get('state_dict',
                                      checkpoint))
        model.load_state_dict(state_dict)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    trainer = Trainer(model, optimizer, device, log_dir=args.log_dir)

    # Train
    trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=args.epochs,
        checkpoint_dir=args.checkpoint_dir
    )
    last_checkpoint_path = os.path.join(args.checkpoint_dir, f"last_checkpoint_model.pt")
    torch.save({'model_state': model.state_dict()}, last_checkpoint_path)

if __name__ == "__main__":
    main()