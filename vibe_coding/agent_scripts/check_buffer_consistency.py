import sys
import os
import torch
import numpy as np

# Add project src folder to PYTHONPATH
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

def check_consistency():
    checkpoint_dir = 'data/checkpoints_v5'
    buffer_path = os.path.join(checkpoint_dir, 'replay_buffer.pt')
    
    print(f"Loading buffer from {buffer_path}...")
    try:
        replay_buffer = torch.load(buffer_path, weights_only=False)
    except Exception as e:
        print(f"Failed to load buffer: {e}")
        return

    print(f"Buffer size: {len(replay_buffer)}")
    
    inconsistent_count = 0
    
    max_factory_val = -1e9
    
    for i, ex in enumerate(replay_buffer):
        obs = np.array(ex['obs'])
        
        # Check for NaN/Inf in obs
        if np.isnan(obs).any() or np.isinf(obs).any():
            inconsistent_count += 1
            print(f"Ex {i}: NaN/Inf in observation!")
            continue
            
        pi = np.array(ex['pi'])
        v = np.array(ex['v'])
        if np.isnan(pi).any() or np.isinf(pi).any():
            print(f"Ex {i}: NaN/Inf in pi!")
            inconsistent_count += 1
        if np.isnan(v).any() or np.isinf(v).any():
            print(f"Ex {i}: NaN/Inf in v!")
            
        # Check factories part
        # Layout: [Spatial (500) | Factories (30) | Global (Rest)]
        spatial_size = 500
        factories_size = 30
        factories = obs[spatial_size : spatial_size + factories_size]
        
        curr_max = factories.max()
        if curr_max > max_factory_val:
            max_factory_val = curr_max
            
        if curr_max > 100: # Threshold for suspicion
             print(f"Ex {i}: Suspicious factory value: {curr_max}")

    print(f"Max factory value found: {max_factory_val}")
    print(f"Total corrupted examples: {inconsistent_count}")

    
    print(f"Total inconsistent examples: {inconsistent_count}")

if __name__ == "__main__":
    check_consistency()
