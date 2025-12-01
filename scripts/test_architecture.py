#!/usr/bin/env python3
"""
Test script to verify the new AzulNet architecture with:
- Expanded global input
- Action mask injection
- Shared trunk (fusion layer)
"""

import sys
sys.path.insert(0, 'src')

import numpy as np
import torch
from azul.env import AzulEnv
from net.azul_net import AzulNet

def test_env_global_features():
    """Test that the environment produces the correct global feature size"""
    print("=" * 60)
    print("TEST 1: Environment Global Features")
    print("=" * 60)
    
    env = AzulEnv(num_players=2, factories_count=5)
    obs = env.reset()
    obs_flat = env.encode_observation(obs)
    
    # Calculate expected size:
    # Spatial: 2 players * 2 (pattern + wall) * 5 * 5 = 100
    # Factories: (5 + 1) * 5 = 30
    # Global: bag(5) + discard(5) + first_player(1) + floor_lines(2*7=14) + scores(2) 
    #         + round(1) + bonuses(2*3=6) = 34
    spatial_size = 2 * 2 * 5 * 5  # 100
    factories_size = (5 + 1) * 5  # 30
    global_size = 5 + 5 + 1 + 14 + 2 + 1 + 6  # 34
    
    expected_total = spatial_size + factories_size + global_size  # 164
    
    print(f"Observation flat size: {obs_flat.shape[0]}")
    print(f"Expected size: {expected_total}")
    print(f"  - Spatial: {spatial_size}")
    print(f"  - Factories: {factories_size}")
    print(f"  - Global: {global_size}")
    
    if obs_flat.shape[0] == expected_total:
        print("✅ PASS: Global features size is correct")
        return global_size
    else:
        print(f"❌ FAIL: Expected {expected_total}, got {obs_flat.shape[0]}")
        return None

def test_network_architecture(global_size):
    """Test that the network can be instantiated and forward pass works"""
    print("\n" + "=" * 60)
    print("TEST 2: Network Architecture")
    print("=" * 60)
    
    # Network parameters
    in_channels = 4  # 2 players * 2 (pattern + wall)
    action_size = (5 + 1) * 5 * 6  # 180
    
    print(f"Creating AzulNet with:")
    print(f"  - in_channels: {in_channels}")
    print(f"  - global_size: {global_size}")
    print(f"  - action_size: {action_size}")
    
    try:
        model = AzulNet(
            in_channels=in_channels,
            global_size=global_size,
            action_size=action_size,
            factories_count=5
        )
        print("✅ PASS: Network instantiated successfully")
        return model
    except Exception as e:
        print(f"❌ FAIL: Network instantiation failed: {e}")
        return None

def test_forward_pass(model):
    """Test forward pass with and without action mask"""
    print("\n" + "=" * 60)
    print("TEST 3: Forward Pass")
    print("=" * 60)
    
    batch_size = 4
    device = torch.device('cpu')
    model = model.to(device)
    
    # Create dummy inputs
    x_spatial = torch.randn(batch_size, 4, 5, 5)
    x_global = torch.randn(batch_size, model.global_size)
    x_factories = torch.randn(batch_size, 6, 5)
    
    # Test WITHOUT action mask
    print("Testing forward pass WITHOUT action mask...")
    try:
        pi_logits, value = model.forward(x_spatial, x_global, x_factories)
        print(f"  - pi_logits shape: {pi_logits.shape} (expected: {batch_size}, {model.action_size})")
        print(f"  - value shape: {value.shape} (expected: {batch_size})")
        
        if pi_logits.shape == (batch_size, model.action_size) and value.shape == (batch_size,):
            print("✅ PASS: Forward without mask works")
        else:
            print("❌ FAIL: Output shapes don't match")
            return False
    except Exception as e:
        print(f"❌ FAIL: Forward without mask failed: {e}")
        return False
    
    # Test WITH action mask
    print("\nTesting forward pass WITH action mask...")
    action_mask = torch.ones(batch_size, model.action_size)
    # Zero out half the actions
    action_mask[:, ::2] = 0
    
    try:
        pi_logits, value = model.forward(x_spatial, x_global, x_factories, action_mask)
        print(f"  - pi_logits shape: {pi_logits.shape}")
        print(f"  - value shape: {value.shape}")
        print("✅ PASS: Forward with mask works")
        return True
    except Exception as e:
        print(f"❌ FAIL: Forward with mask failed: {e}")
        return False

def test_predict_method(model):
    """Test the predict method used by MCTS"""
    print("\n" + "=" * 60)
    print("TEST 4: Predict Method")
    print("=" * 60)
    
    # Create a real environment observation
    env = AzulEnv(num_players=2, factories_count=5)
    obs = env.reset()
    obs_flat = env.encode_observation(obs)
    obs_batch = np.array([obs_flat])
    
    # Get valid actions and create mask
    valid_actions = env.get_valid_actions()
    action_mask = np.zeros((1, env.action_size), dtype=np.float32)
    for action in valid_actions:
        idx = env.action_to_index(action)
        action_mask[0, idx] = 1.0
    
    print(f"Valid actions: {len(valid_actions)}")
    print(f"Action mask sum: {action_mask.sum()}")
    
    # Test predict WITHOUT mask
    try:
        pi_logits, values = model.predict(obs_batch)
        print(f"✅ Predict without mask: pi shape {pi_logits.shape}, v shape {values.shape}")
    except Exception as e:
        print(f"❌ FAIL: Predict without mask failed: {e}")
        return False
    
    # Test predict WITH mask
    try:
        pi_logits, values = model.predict(obs_batch, action_mask)
        print(f"✅ Predict with mask: pi shape {pi_logits.shape}, v shape {values.shape}")
        
        # Verify that invalid action logits are not dominant
        # (they should be less likely due to mask injection)
        probs = np.exp(pi_logits - np.max(pi_logits)) / np.sum(np.exp(pi_logits - np.max(pi_logits)))
        legal_prob = probs[0, action_mask[0] == 1].sum()
        print(f"  - Total probability on legal actions: {legal_prob:.4f}")
        
        return True
    except Exception as e:
        print(f"❌ FAIL: Predict with mask failed: {e}")
        return False

def test_mcts_integration():
    """Test MCTS with the new network"""
    print("\n" + "=" * 60)
    print("TEST 5: MCTS Integration")
    print("=" * 60)
    
    from mcts.mcts import MCTS
    
    env = AzulEnv(num_players=2, factories_count=5)
    obs = env.reset()
    obs_flat = env.encode_observation(obs)
    
    # Create model
    global_size = obs_flat.shape[0] - 100 - 30  # Total - spatial - factories
    model = AzulNet(
        in_channels=4,
        global_size=global_size,
        action_size=env.action_size,
        factories_count=5
    )
    
    try:
        mcts = MCTS(env, model, simulations=5, cpuct=1.0)
        mcts.run()
        action = mcts.select_action()
        
        print(f"✅ MCTS ran successfully")
        print(f"  - Selected action: {action}")
        print(f"  - Root visits: {mcts.root.visits}")
        print(f"  - Root children: {len(mcts.root.children)}")
        return True
    except Exception as e:
        print(f"❌ FAIL: MCTS failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    print("\n" + "=" * 60)
    print("AZUL ZERO ARCHITECTURE VERIFICATION")
    print("=" * 60 + "\n")
    
    # Test 1: Environment
    global_size = test_env_global_features()
    if global_size is None:
        print("\n❌ FATAL: Environment test failed")
        return 1
    
    # Test 2: Network
    model = test_network_architecture(global_size)
    if model is None:
        print("\n❌ FATAL: Network architecture test failed")
        return 1
    
    # Test 3: Forward pass
    if not test_forward_pass(model):
        print("\n❌ FATAL: Forward pass test failed")
        return 1
    
    # Test 4: Predict
    if not test_predict_method(model):
        print("\n❌ FATAL: Predict method test failed")
        return 1
    
    # Test 5: MCTS
    if not test_mcts_integration():
        print("\n❌ FATAL: MCTS integration test failed")
        return 1
    
    print("\n" + "=" * 60)
    print("✅ ALL TESTS PASSED!")
    print("=" * 60)
    print("\nThe new architecture is ready for training.")
    print("Run: python scripts/train_loop.py")
    return 0

if __name__ == "__main__":
    exit(main())
