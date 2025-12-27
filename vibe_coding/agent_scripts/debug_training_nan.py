import sys
import os
import torch
from torch.utils.data import DataLoader

# Add project src folder to PYTHONPATH
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from net.azul_net import AzulNet
from train.dataset import AzulDataset
from train.trainer import Trainer

def debug_nan():
    # Paths
    checkpoint_dir = 'data/checkpoints_v5'
    model_path = os.path.join(checkpoint_dir, 'model_cycle_14.pt')
    buffer_path = os.path.join(checkpoint_dir, 'replay_buffer.pt')
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load Replay Buffer
    print(f"Loading buffer from {buffer_path}...")
    try:
        replay_buffer = torch.load(buffer_path, weights_only=False) # Allow all classes for now or use safe globals
    except Exception as e:
        print(f"Failed to load buffer: {e}")
        return

    print(f"Buffer size: {len(replay_buffer)}")
    if len(replay_buffer) > 0:
        print(f"Keys in first example: {replay_buffer[0].keys()}")

    
    # Dataset
    dataset = AzulDataset(replay_buffer, augment_factories=True)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
    
    # Load Model
    print(f"Loading model from {model_path}...")
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    
    # Params
    params = checkpoint.get('params', {})
    print(f"Params: {params}")

    model = AzulNet(
        in_channels=20, # Hardcoded based on code analysis
        global_size=46, # Derived from logs: Total=576, Spatial=500, Factories=30 -> Global=46
        action_size=180, # Correct size: (5+1)*5*6
        factories_count=5
    ).to(device)
    
    model.load_state_dict(checkpoint['model_state'])
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    if 'optimizer_state' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state'])

    trainer = Trainer(model, optimizer, device)
    
    print("Starting debug training epoch with anomaly detection...")
    
    # Enable Anomaly Detection
    torch.autograd.set_detect_anomaly(True)
    
    try:
        # We manually run one epoch calling train_epoch logic or just loop here
        # Let's trust trainer.fit but we need to verify if trainer.py was already modified to include anomaly detection?
        # No, I didn't modify it yet. I will rely on the context manager here if I run the loop manually.
        # But Trainer.train_epoch doesn't look at the context manager? Yes it does, it's global.
        
        # But wait, Trainer catches nothing. So let's just call trainer.train_epoch
        trainer.train_epoch(dataloader, epoch=1)
        
    except RuntimeError as e:
        print("\n\nAnalyzed Runtime Error (likely Anomaly):")
        print(e)
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_nan()
