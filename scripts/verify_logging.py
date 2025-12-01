import sys
import os
import shutil

# Add project src folder to PYTHONPATH
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

# Add scripts folder to PYTHONPATH
sys.path.insert(0, os.path.dirname(__file__))
import train_loop

def mock_get_curriculum_params(cycle):
    return {
        'n_games': 1,
        'simulations': 2,
        'epochs': 1,
        'lr': 1e-3,
        'cpuct': 1.0
    }

def verify():
    print("Verifying Logging...")
    
    # Clean up logs dir
    if os.path.exists("logs"):
        shutil.rmtree("logs")
    
    # Monkeypatch
    train_loop.get_curriculum_params = mock_get_curriculum_params
    
    # Run main with modified args
    # We need to mock sys.argv or modify main to accept args
    # train_loop.main() parses args.
    sys.argv = ["train_loop.py", "--total_cycles", "1", "--checkpoint_dir", "data/checkpoints_test"]
    
    try:
        train_loop.main()
    except SystemExit:
        pass
    except Exception as e:
        print(f"❌ Training loop failed: {e}")
        return False
        
    # Check logs
    if not os.path.exists("logs/training.log"):
        print("❌ logs/training.log not found")
        return False
    
    if not os.path.exists("logs/training_monitor.log"):
        print("❌ logs/training_monitor.log not found")
        return False
        
    with open("logs/training.log", "r") as f:
        content = f.read()
        if "Training Session Started" not in content:
            print("❌ Log missing start message")
            return False
        if "Cycle 1/1" not in content:
            print("❌ Log missing cycle info")
            return False
            
    print("✅ Logging Verified!")
    return True

if __name__ == "__main__":
    if not verify():
        exit(1)
