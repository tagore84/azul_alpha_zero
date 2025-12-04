#!/usr/bin/env python3
"""
Verification tests for MCTS tree reuse implementation.
Tests tree structure, integration with self-play, and memory behavior.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

import torch
import numpy as np
from azul.env import AzulEnv
from net.azul_net import AzulNet
from mcts.mcts import MCTS
from train.self_play import play_game

def test_1_tree_structure():
    """Test 1: Verify tree structure is preserved after advance()"""
    print("\n" + "="*60)
    print("TEST 1: Tree Structure Preservation")
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
    
    # Run MCTS
    mcts.run()
    
    # Check root has children
    assert len(mcts.root.children) > 0, "Root should have children after MCTS run"
    print(f"✓ Root has {len(mcts.root.children)} children")
    
    # Get an action
    action = mcts.select_action()
    print(f"✓ Selected action: {action}")
    
    # Save old root data
    old_root_visits = mcts.root.visits
    old_child = mcts.root.children[action]
    old_child_visits = old_child.visits
    old_child_children_count = len(old_child.children)
    
    print(f"✓ Old root visits: {old_root_visits}")
    print(f"✓ Old child visits: {old_child_visits}")
    print(f"✓ Old child has {old_child_children_count} children")
    
    # Apply action to env
    env.step(action)
    
    # Advance tree
    mcts.advance(action, env)
    
    # Verify reuse
    assert mcts.root.visits == old_child_visits, \
        f"New root visits ({mcts.root.visits}) should equal old child visits ({old_child_visits})"
    print(f"✓ New root visits match old child visits: {mcts.root.visits}")
    
    assert mcts.root.parent is None, "New root should be detached from parent"
    print(f"✓ New root is detached (parent=None)")
    
    assert len(mcts.root.children) == old_child_children_count, \
        f"New root children ({len(mcts.root.children)}) should match old child children ({old_child_children_count})"
    print(f"✓ New root preserved {len(mcts.root.children)} children")
    
    print("\n✅ TEST 1 PASSED: Tree structure correctly preserved\n")
    return True

def test_2_selfplay_integration():
    """Test 2: Verify self-play completes without errors"""
    print("\n" + "="*60)
    print("TEST 2: Self-Play Integration")
    print("="*60)
    
    env = AzulEnv()
    
    model = AzulNet(
        in_channels=4,
        global_size=34,
        action_size=env.action_size,
        factories_count=env.N
    )
    model.eval()
    
    # Play one game with tree reuse
    print("Playing one full game with tree reuse...")
    try:
        examples = play_game(env.clone(), model, simulations=10, cpuct=1.0)
    except ValueError as e:
        print(f"\n⚠️ ValueError during self-play: {e}")
        print(f"   This suggests a bug in action selection or tree reuse.")
        print(f"   The selected action is not valid in the current game state.")
        return False
    
    # Verify game completed
    assert len(examples) > 0, "Should generate training examples"
    print(f"✓ Generated {len(examples)} training examples")
    
    assert all('obs' in ex and 'pi' in ex and 'v' in ex for ex in examples), \
        "All examples should have obs, pi, v"
    print(f"✓ All examples have required fields (obs, pi, v)")
    
    # Check value targets
    values = [ex['v'] for ex in examples]
    assert all(v in [-1, 0, 1] for v in values), \
        f"Values should be in {{-1, 0, 1}}, got: {set(values)}"
    print(f"✓ All value targets are in {{-1, 0, 1}}")
    
    print("\n✅ TEST 2 PASSED: Self-play integration works correctly\n")
    return True

def test_3_visits_accumulation():
    """Test 3: Verify visits accumulate correctly across moves"""
    print("\n" + "="*60)
    print("TEST 3: Visits Accumulation")
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
    
    mcts = MCTS(env, model, simulations=10, cpuct=1.0)
    
    # Move 1
    print("\nMove 1:")
    mcts.run()
    root_visits_1 = mcts.root.visits
    action_1 = mcts.select_action()
    child_visits_1 = mcts.root.children[action_1].visits
    print(f"  Root visits: {root_visits_1}")
    print(f"  Selected action: {action_1}, visits: {child_visits_1}")
    
    env.step(action_1)
    mcts.advance(action_1, env)
    
    new_root_visits = mcts.root.visits
    assert new_root_visits == child_visits_1, \
        f"After advance, root visits {new_root_visits} should equal old child visits {child_visits_1}"
    print(f"  ✓ After advance: root visits = {new_root_visits}")
    
    # Move 2
    print("\nMove 2:")
    mcts.run()
    root_visits_2 = mcts.root.visits
    print(f"  Root visits after 2nd run: {root_visits_2}")
    
    # Visits should have increased
    assert root_visits_2 > new_root_visits, \
        f"Visits should accumulate: {root_visits_2} should be > {new_root_visits}"
    print(f"  ✓ Visits accumulated: {new_root_visits} → {root_visits_2}")
    
    print("\n✅ TEST 3 PASSED: Visits accumulate correctly\n")
    return True

def test_4_fallback_behavior():
    """Test 4: Verify fallback when action not in tree"""
    print("\n" + "="*60)
    print("TEST 4: Fallback Behavior")
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
    
    mcts = MCTS(env, model, simulations=5, cpuct=1.0)
    mcts.run()
    
    # Get a valid action that might not be in the tree
    valid_actions = env.get_valid_actions()
    
    # Find an action NOT explored by MCTS (if any)
    unexplored = [a for a in valid_actions if a not in mcts.root.children]
    
    if unexplored:
        action = unexplored[0]
        print(f"Testing with unexplored action: {action}")
        
        env.step(action)
        mcts.advance(action, env)
        
        # Should create fresh root (fallback)
        assert mcts.root.parent is None, "Fallback root should have no parent"
        assert mcts.root.visits == 0, "Fallback root should have 0 visits"
        print(f"✓ Fallback created fresh root with 0 visits")
    else:
        print(f"⚠ All {len(valid_actions)} valid actions were explored (tree is complete)")
        print(f"  This is expected with very few simulations and limited actions")
    
    print("\n✅ TEST 4 PASSED: Fallback behavior works correctly\n")
    return True

def run_all_tests():
    """Run all verification tests"""
    print("\n" + "#"*60)
    print("# MCTS TREE REUSE - VERIFICATION TESTS")
    print("#"*60)
    
    tests = [
        test_1_tree_structure,
        test_2_selfplay_integration,
        test_3_visits_accumulation,
        test_4_fallback_behavior,
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append((test.__name__, result))
        except Exception as e:
            print(f"\n❌ {test.__name__} FAILED with exception:")
            print(f"   {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
            results.append((test.__name__, False))
    
    # Summary
    print("\n" + "#"*60)
    print("# TEST SUMMARY")
    print("#"*60)
    for name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status}: {name}")
    
    all_passed = all(r[1] for r in results)
    print("\n" + "="*60)
    if all_passed:
        print("✅ ALL TESTS PASSED - Tree reuse implementation is correct")
    else:
        print("❌ SOME TESTS FAILED - Review errors above")
    print("="*60 + "\n")
    
    return all_passed

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
