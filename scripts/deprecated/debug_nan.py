
import torch
import torch.nn as nn
import sys
import os

# Add project src folder to PYTHONPATH
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from net.azul_net import AzulNet

def test_training_step(device_name='cpu'):
    print(f"Testing on device: {device_name}")
    if device_name == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available")
        return
    if device_name == 'mps' and not torch.backends.mps.is_available():
        print("MPS not available")
        return
        
    device = torch.device(device_name)
    
    # Dimensions based on logs
    # Shapes: Spatial=100, Factories=30, Global=39, Total=169
    in_channels = 4
    spatial_size = 100 # 4*5*5
    factories_count = 5
    factories_size = (factories_count + 1) * 5 # 30
    global_size = 39
    action_size = 100 # Approx? Need exact. Let's assume standard Azul.
    # Env action size from code?
    # azul/env.py is needed to get exact action size but we can guess or import
    from azul.env import AzulEnv
    env = AzulEnv()
    action_size = env.action_size
    print(f"Action Size: {action_size}")
    
    model = AzulNet(
        in_channels=in_channels,
        global_size=global_size,
        action_size=action_size,
        factories_count=factories_count
    ).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    batch_size = 64
    
    # Create random batch
    obs_spatial = torch.randn(batch_size, in_channels, 5, 5).to(device)
    obs_factories = torch.randn(batch_size, factories_count+1, 5).to(device)
    obs_global = torch.randn(batch_size, global_size).to(device)
    
    target_pi = torch.softmax(torch.randn(batch_size, action_size), dim=1).to(device)
    target_v = torch.tanh(torch.randn(batch_size, 1)).to(device)
    
    print("Initial parameters check:")
    for name, param in model.named_parameters():
        if torch.isnan(param).any():
            print(f"NaN in {name} before training")
            
    # Train Step
    model.train()
    optimizer.zero_grad()
    
    pi_logits, value = model(obs_spatial, obs_global, obs_factories)
    
    if torch.isnan(pi_logits).any():
        print("NaN in pi_logits output")
    if torch.isnan(value).any():
        print("NaN in value output")
        
    log_pi = torch.nn.functional.log_softmax(pi_logits, dim=1)
    l_pi = -(target_pi * log_pi).sum(dim=1).mean()
    l_v = torch.nn.functional.mse_loss(value, target_v)
    loss = l_pi + l_v
    
    print(f"Loss: {loss.item()}")
    
    loss.backward()
    
    print("Gradients check:")
    has_nan_grad = False
    for name, param in model.named_parameters():
        if param.grad is not None and torch.isnan(param.grad).any():
            print(f"NaN gradient in {name}")
            has_nan_grad = True
            
    if not has_nan_grad:
        print("No NaN gradients found.")
        
    optimizer.step()
    
    print("Parameters check after step:")
    for name, param in model.named_parameters():
        if torch.isnan(param).any():
            print(f"NaN in {name} after step")

if __name__ == "__main__":
    print("--- Testing CPU ---")
    test_training_step('cpu')
    print("\n--- Testing MPS ---")
    test_training_step('mps')
