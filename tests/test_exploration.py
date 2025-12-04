#!/usr/bin/env python3
"""
Verification tests for Exploration Improvements.
Tests temperature sampling and Dirichlet noise.
"""

import sys
import os
import numpy as np
from collections import Counter

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from azul.env import AzulEnv
from net.azul_net import AzulNet
from mcts.mcts import MCTS

def test_temperature_sampling():
    """Test 1: Verify temperature affects action selection"""
    print("\n" + "="*60)
    print("TEST 1: Temperature Sampling")
    print("="*60)
    
    env = AzulEnv()
    env.reset()
    
    # Create minimal model
    model = AzulNet(
        in_channels=4,
        global_size=34,
        action_size=env.action_size,
        factories_count=env.N
    )
    model.eval()
    
    mcts = MCTS(env, model, simulations=10, cpuct=1.0)
    mcts.run()
    
    # Manually set visit counts to create a clear distribution
    # Action A: 100 visits
    # Action B: 10 visits
    children = list(mcts.root.children.items())
    if len(children) < 2:
        print("⚠ Not enough children to test sampling")
        return False
        
    action_a, node_a = children[0]
    action_b, node_b = children[1]
    
    node_a.visits = 100
    node_b.visits = 10
    
    print(f"Action A visits: {node_a.visits}")
    print(f"Action B visits: {node_b.visits}")
    
    # Test Temp=0 (Greedy)
    print("\nTesting Temp=0 (Greedy):")
    actions_greedy = [mcts.select_action(temperature=0) for _ in range(20)]
    counts_greedy = Counter(actions_greedy)
    print(f"  Selection counts: {counts_greedy}")
    
    assert counts_greedy[action_a] == 20, "Temp=0 should always select max visited action"
    assert counts_greedy[action_b] == 0, "Temp=0 should never select lower visited action"
    print("✓ Temp=0 is deterministic and greedy")
    
    # Test Temp=1 (Stochastic)
    print("\nTesting Temp=1 (Stochastic):")
    actions_temp1 = [mcts.select_action(temperature=1.0) for _ in range(1000)]
    counts_temp1 = Counter(actions_temp1)
    ratio = counts_temp1[action_a] / counts_temp1[action_b]
    expected_ratio = 100 / 10
    print(f"  Selection counts: A={counts_temp1[action_a]}, B={counts_temp1[action_b]}")
    print(f"  Ratio A/B: {ratio:.2f} (Expected ~{expected_ratio:.2f})")
    
    assert counts_temp1[action_b] > 0, "Temp=1 should sometimes select lower visited action"
    print("✓ Temp=1 is stochastic")
    
    print("\n✅ TEST 1 PASSED: Temperature sampling works correctly\n")
    return True

def test_dirichlet_noise():
    """Test 2: Verify Dirichlet noise changes priors"""
    print("\n" + "="*60)
    print("TEST 2: Dirichlet Noise")
    print("="*60)
    
    env = AzulEnv()
    env.reset()
    
    model = AzulNet(
        in_channels=4,
        global_size=34,
        action_size=env.action_size,
        factories_count=env.N
    )
    model.eval()
    
    mcts = MCTS(env, model, simulations=2, cpuct=1.0)
    mcts.expand(mcts.root)
    
    # Get initial priors
    initial_priors = {a: n.prior for a, n in mcts.root.children.items()}
    print(f"Initial priors (first 3): {list(initial_priors.values())[:3]}")
    
    # Add noise
    epsilon = 0.5 # High epsilon to make change obvious
    mcts.add_root_noise(alpha=0.3, epsilon=epsilon)
    
    # Get new priors
    new_priors = {a: n.prior for a, n in mcts.root.children.items()}
    print(f"New priors (first 3):     {list(new_priors.values())[:3]}")
    
    # Verify changes
    changed = False
    for a in initial_priors:
        if abs(initial_priors[a] - new_priors[a]) > 1e-6:
            changed = True
            break
            
    assert changed, "Priors should change after adding noise"
    print("✓ Priors changed after adding noise")
    
    # Verify sum is still ~1 (Dirichlet sums to 1, priors sum to 1)
    total_prior = sum(new_priors.values())
    print(f"Total prior sum: {total_prior:.6f}")
    assert abs(total_prior - 1.0) < 1e-2, f"Priors should sum to 1.0, got {total_prior}"
    print("✓ Priors still sum to ~1.0")
    
    print("\n✅ TEST 2 PASSED: Dirichlet noise applied correctly\n")
    return True

if __name__ == "__main__":
    t1 = test_temperature_sampling()
    t2 = test_dirichlet_noise()
    
    if t1 and t2:
        print("✅ ALL EXPLORATION TESTS PASSED")
        sys.exit(0)
    else:
        print("❌ SOME TESTS FAILED")
        sys.exit(1)
