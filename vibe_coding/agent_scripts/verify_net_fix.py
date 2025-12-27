
import sys
import os
import torch
import numpy as np

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from net.azul_net import AzulNet

def verify_net():
    print("Verifying AzulNet structure and forward pass...")
    
    # Parameters
    in_channels = 6  # Example
    global_size = 12 # Example
    action_size = 100 # Example
    factories_count = 5
    
    model = AzulNet(in_channels, global_size, action_size, factories_count=factories_count)
    model.eval()
    
    # Dummy input
    batch_size = 2
    x_spatial = torch.randn(batch_size, in_channels, 5, 5)
    x_global = torch.randn(batch_size, global_size)
    x_factories = torch.randn(batch_size, factories_count + 1, 5)
    action_mask = torch.ones(batch_size, action_size)
    
    print(f"Input shapes: Spatial {x_spatial.shape}, Global {x_global.shape}, Factories {x_factories.shape}")
    
    try:
        pi, v = model(x_spatial, x_global, x_factories, action_mask)
        print(f"Output shapes: Policy {pi.shape}, Value {v.shape}")
        
        # Verify shapes
        assert pi.shape == (batch_size, action_size)
        assert v.shape == (batch_size,)
        
        print("SUCCESS: Forward pass complete and shapes are correct.")
    except Exception as e:
        print(f"FAILURE: Forward pass failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    verify_net()
