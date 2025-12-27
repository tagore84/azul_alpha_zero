
import torch
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from net.azul_net import AzulNet

def check_checkpoint(path):
    print(f"Checking {path}...")
    if not os.path.exists(path):
        print("File not found.")
        return

    try:
        ckpt = torch.load(path, map_location='cpu')
        state_dict = ckpt['model_state']
        
        has_nan = False
        for key, val in state_dict.items():
            if torch.isnan(val).any():
                print(f"!! NaN found in {key}")
                has_nan = True
            if torch.isinf(val).any():
                print(f"!! Inf found in {key}")
                has_nan = True
                
        if not has_nan:
            print("Model weights look clean (no NaNs/Infs).")
        else:
            print("Model is corrupted.")
            
    except Exception as e:
        print(f"Error loading: {e}")

if __name__ == "__main__":
    check_checkpoint("data/checkpoints_v5/model_cycle_1.pt")
    check_checkpoint("data/checkpoints_v5/model_cycle_2.pt")
