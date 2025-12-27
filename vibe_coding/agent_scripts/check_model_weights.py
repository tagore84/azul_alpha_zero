
import torch
import sys

def check_weights(path):
    print(f"Checking {path}...")
    try:
        checkpoint = torch.load(path, map_location='cpu')
        model_state = checkpoint.get('model_state', checkpoint)
        
        has_nan = False
        for name, param in model_state.items():
            if torch.isnan(param).any():
                print(f"❌ Layer {name} contains NaNs!")
                has_nan = True
            if torch.isinf(param).any():
                print(f"❌ Layer {name} contains Infs!")
                has_nan = True
                
        if not has_nan:
            print("✅ No NaNs or Infs found in model weights.")
        else:
            print("❌ Model is corrupted.")
            
    except Exception as e:
        print(f"Error loading {path}: {e}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python check_model_weights.py <path_to_model>")
        sys.exit(1)
    check_weights(sys.argv[1])
