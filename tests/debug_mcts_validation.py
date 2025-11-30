
import torch
import numpy as np
from azul.env import AzulEnv
from mcts.mcts import MCTS
from net.azul_net import AzulNet

def test_mcts_expansion():
    print("Initializing Env...")
    env = AzulEnv()
    env.reset()
    
    print("Initializing Model...")
    # Create a dummy model
    # Correct sizes based on env.py analysis:
    # Spatial: 4 channels (2 players * (1 pattern + 1 wall))
    # Global: 57
    model = AzulNet(in_channels=4, global_size=57, action_size=env.action_size)
    model.eval()
    
    print("Initializing MCTS...")
    mcts = MCTS(env, model, simulations=25, cpuct=1.0)
    
    print("Running MCTS...")
    mcts.run()
    
    print(f"Root children count: {len(mcts.root.children)}")
    
    if not mcts.root.children:
        print("FAIL: No children found.")
        # Debug why
        print(f"Valid actions from root env: {len(mcts.root.env.get_valid_actions())}")
        
        # Check done status
        done = any(all(cell != -1 for cell in row) for p in mcts.root.env.players for row in p['wall'])
        print(f"Is root env done? {done}")
    else:
        print("SUCCESS: Children found.")
        action = mcts.select_action()
        print(f"Selected action: {action}")

if __name__ == "__main__":
    test_mcts_expansion()
