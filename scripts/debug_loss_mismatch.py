
import sys
import os
import torch
import numpy as np

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from azul.env import AzulEnv
from net.azul_net import AzulNet
from mcts.mcts import MCTS


import sys
import os
import torch
import numpy as np
import time

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from azul.env import AzulEnv
from net.azul_net import AzulNet
from mcts.mcts import MCTS

def debug_game_mismatch():
    print("Initializing Debug Session for Full Game...")
    env = AzulEnv(num_players=2)
    env.reset()
    
    # Init random model
    in_channels = 4
    spatial_size = 100
    factories_size = 30
    obs_test = env.encode_observation(env._get_obs())
    global_size = obs_test.shape[0] - spatial_size - factories_size
    action_size = env.action_size
    print(f"Action Size: {action_size}")
    
    model = AzulNet(in_channels, global_size, action_size, factories_count=env.N)
    model.eval()
    
    mcts = MCTS(env, model, simulations=50, cpuct=1.0)
    
    done = False
    moves = 0
    mismatch_count = 0
    
    while not done:
        moves += 1
        # 1. Capture Mask
        action_mask = env.get_action_mask()
        
        # 2. Run MCTS
        mcts.run() # uses internal root env which should be synced
        
        # 3. Get Targets
        root = mcts.root
        visits = np.zeros(action_size, dtype=np.float32)
        valid_actions_mcts = []
        for action, node in root.children.items():
            idx = env.action_to_index(action)
            visits[idx] = node.visits
            valid_actions_mcts.append(idx)
        
        if visits.sum() == 0:
            print("WARNING: No visits collected?")
            break
            
        pi_target = visits / visits.sum()
        
        # 4. Check Consistency
        # Mismatch if target > 0 but mask == 0
        local_mismatch = []
        for i in range(action_size):
            if pi_target[i] > 0 and action_mask[i] == 0:
                local_mismatch.append((i, pi_target[i]))
        
        if local_mismatch:
            print(f"CRITICAL: Move {moves} - Found {len(local_mismatch)} mismatches!")
            mismatch_count += 1
            for idx, prob in local_mismatch:
                print(f"  Index {idx}: MCTS Prob {prob:.4f}, Mask {action_mask[idx]}")
                print(f"  Action: {env.index_to_action(idx)}")
            
            # Print Environment states to see if they differ
            print("  Main Env Valid Actions:", env.get_valid_actions())
            print("  MCTS Root Env Valid Actions:", root.env.get_valid_actions())
            
        # 5. Check Loss
        obs = env.encode_observation(env._get_obs())
        obs_tensor = torch.tensor(obs).unsqueeze(0).float()
        mask_tensor = torch.tensor(action_mask).unsqueeze(0).float()
        target_tensor = torch.tensor(pi_target).unsqueeze(0).float()
        
        spatial = obs_tensor[:, :100].view(1, 4, 5, 5)
        factories = obs_tensor[:, 100:130].view(1, 6, 5)
        glob = obs_tensor[:, 130:]
        
        pi_logits, val = model(spatial, glob, factories, action_mask=mask_tensor)
        log_pi = torch.nn.functional.log_softmax(pi_logits, dim=1)
        # Handle nan/inf manually
        loss_val = -(target_tensor * log_pi).sum(dim=1).mean().item()
        
        if loss_val > 100:
             print(f"  HIGH LOSS: {loss_val:.2f} at move {moves}")
             
        # Step
        action = mcts.select_action(temperature=1.0)
        env.step(action)
        mcts.advance(action, env)
        
        if env.done:
            done = True
            
    if mismatch_count == 0:
        print(f"Success: Game finished in {moves} moves with 0 mismatches.")
    else:
        print(f"Failure: Game finished with {mismatch_count} mismatch events.")
        
if __name__ == "__main__":
    debug_game_mismatch()

if __name__ == "__main__":
    debug_mismatch()
