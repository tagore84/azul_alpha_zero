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
    parser.add_argument('--batch_size', type=int, default=64, help='Training batch size')
    parser.add_argument('--epochs', type=int, default=20, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--train_ratio', type=float, default=0.9, help='Fraction of data for training')
    parser.add_argument('--log_dir', type=str, default='logs', help='TensorBoard log directory')
    parser.add_argument('--checkpoint_dir', type=str, default='data/checkpoint_dir', help='Directory to save checkpoints')
    parser.add_argument('--base_model', type=str, default=None,
                        help='Path to a model checkpoint to resume training from')
    parser.add_argument('--base_dataset', type=str, default=None, help='Path to a dataset to resume training from')
    parser.add_argument('--last_dataset', type=str, default=None, help='Path to a dataset to resume training from')
    parser.add_argument('--max_dataset_size', type=int, default=50000,
                        help='Maximum number of examples to keep in the training dataset')
    args = parser.parse_args()

    base_model = None
    base_dataset = None
    last_dataset = None
    if args.base_model:
        base_model = args.base_model
    if args.base_dataset:
        base_dataset = args.base_dataset
    if args.last_dataset:
        last_dataset = args.last_dataset
        
    # Select device
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    print(f"[train-model] Using device: {device}")

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
    print(f"[train-model] Obs total size: {total_obs_size}, spatial_size: {spatial_size}, global_size: {global_size}, in_channels: {in_channels}")
    action_size = env.action_size

    model = AzulNet(
        in_channels=in_channels,
        global_size=global_size,
        action_size=action_size
    )
    model = model.to(device)
    if base_model:
        checkpoint = torch.load(base_model, map_location=device)
        print(f"[train-model] Loaded base model: {base_model}")
        state_dict = checkpoint.get('model_state',
                       checkpoint.get('state_dict', checkpoint))
        model.load_state_dict(state_dict)
        torch.save({'model_state': model.state_dict()}, base_model.replace('.pt', '_prev.pt'))
    print(f"[train-model] Model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    trainer = Trainer(model, optimizer, device, log_dir=args.log_dir)
    
    new_examples_data = torch.load(last_dataset, weights_only=False)
    new_examples = new_examples_data['examples']
    if not isinstance(new_examples, list):
        raise ValueError("The loaded dataset must contain a list under the 'examples' key")
    print(f"[train-model] Loaded last dataset: {type(new_examples)}, length: {len(new_examples)}")
    if base_dataset:
        historical = torch.load(base_dataset, weights_only=False)
        print(f"[train-model] Loaded base dataset: {type(historical['examples'])}, length: {len(historical['examples'])}")
        random.seed(SEED)  # Usa la misma semilla para consistencia
        if len(new_examples) >= args.max_dataset_size:
            examples = new_examples[-args.max_dataset_size:]
        else:
            num_old_needed = args.max_dataset_size - len(new_examples)
            selected_old_examples = historical['examples'][-num_old_needed:]
            examples = selected_old_examples + new_examples
    else:
        examples = new_examples
        
    shuffled_examples = examples.copy()
    random.shuffle(shuffled_examples)
    
    dataset = AzulDataset(examples.copy(), augment_factories=True)
    train_size = int(len(shuffled_examples) * args.train_ratio)
    val_size = len(shuffled_examples) - train_size
    train_examples = shuffled_examples[:train_size]
    val_examples = shuffled_examples[train_size:]
    train_set = AzulDataset(train_examples, augment_factories=True)
    val_set = AzulDataset(val_examples, augment_factories=False)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size)

    # Print training/validation set sizes
    print(f"[train-model] Training with {len(train_set)} examples, validation with {len(val_set)} examples, total: {len(dataset)}")
    # Train
    trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=args.epochs,
        checkpoint_dir=args.checkpoint_dir
    )
    print(f"[train-model] Training completed. Saving model to {os.path.join(args.checkpoint_dir, 'model_checkpoint.pt')}")
    torch.save({'model_state': model.state_dict()}, os.path.join(args.checkpoint_dir, 'model_checkpoint.pt'))

    # Guardar backup del dataset histórico antes de sobrescribirlo
    if base_dataset and historical:
        backup_path = base_dataset.replace('.pt', '_backup.pt')
        torch.save(historical, backup_path)
        print(f"[train-model] Backup del dataset histórico guardado en: {backup_path}")

        # Combinar ejemplos nuevos e históricos y guardar
        combined_examples = {'examples': examples}
        torch.save(combined_examples, base_dataset)
        print(f"[train-model] Dataset combinado guardado en: {base_dataset}")
    else:
        combined_examples = {'examples': examples}
        torch.save({'examples': examples}, 'data/checkpoint_dir/all_historical_dataset.pt')

if __name__ == "__main__":
    main()