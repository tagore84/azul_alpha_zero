import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from azul.env import AzulEnv

def test_done_reset():
    print("Testing env.done reset bug fix...")
    env = AzulEnv()
    
    # Reset environment
    obs = env.reset(initial=True)
    print(f"After reset: env.done = {env.done}")
    assert env.done == False, "env.done should be False after reset"
    
    # Play until game ends
    max_steps = 1000
    for step in range(max_steps):
        valid_actions = env.get_valid_actions()
        if not valid_actions:
            print(f"No valid actions at step {step}")
            break
        action = valid_actions[0]
        obs, reward, done, info = env.step(action)
        if done:
            print(f"Game ended at step {step}: env.done = {env.done}")
            assert env.done == True, "env.done should be True when game ends"
            break
    
    # Reset again
    obs = env.reset()
    print(f"After second reset: env.done = {env.done}")
    assert env.done == False, "env.done should be False after reset"
    
    print("✅ Test passed!")

if __name__ == "__main__":
    test_done_reset()
