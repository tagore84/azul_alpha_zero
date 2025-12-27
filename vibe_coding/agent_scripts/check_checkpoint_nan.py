
import torch
import sys
import os

def check_checkpoint(path):
    print(f"Checking {path}...")
    if not os.path.exists(path):
        print("File not found.")
        return
    
    try:
        data = torch.load(path, map_location='cpu')
        model_state = data['model_state']
        
        has_nan = False
        for name, param in model_state.items():
            if torch.isnan(param).any():
                print(f"!! parameter {name} has NaNs!")
                has_nan = True
            if torch.isinf(param).any():
                print(f"!! parameter {name} has Infs!")
                has_nan = True
                
        if not has_nan:
            print("Model weights seem CLEAN (no NaNs/Infs).")
        else:
            print("Model weights are CORRUPTED.")
            
    except Exception as e:
        print(f"Error loading: {e}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        check_checkpoint(sys.argv[1])
    else:
        # Check default locations
        base = "data/checkpoints_v5"
        files = [f for f in os.listdir(base) if f.endswith(".pt")]
        files.sort()
        if files:
            print(f"Found {len(files)} checkpoints. Checking latest 3...")
            for f in files[-3:]:
                check_checkpoint(os.path.join(base, f))
        else:
            print(f"No checkpoints found in {base}")
