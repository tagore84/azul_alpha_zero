

import sys
import os

from constants import SEED
# Add project src folder to PYTHONPATH for module resolution
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from azul.env import AzulEnv
from mcts.mcts import MCTS
import numpy as np

class DummyModel:
    """
    Dummy model implementing the .predict interface expected by MCTS.
    """
    def __init__(self, action_size):
        self.action_size = action_size

    def predict(self, obs_batch: np.ndarray):
        batch_size = obs_batch.shape[0]
        # Uniform policy logits over all actions
        pi_logits = np.zeros((batch_size, self.action_size), dtype=float)
        # Neutral value estimate = 0 for all states
        values = np.zeros((batch_size, 1), dtype=float)
        return pi_logits, values

def main():
    # Initialize the Azul environment
    env = AzulEnv(num_players=2, factories_count=5, seed=SEED)

    # Instantiate the dummy model
    model = DummyModel(env.action_size)

    # Create the MCTS object with the environment and model
    mcts = MCTS(env, model=model, simulations=20, cpuct=1.0)

    # Run MCTS simulations to populate the search tree
    mcts.run()

    # Select the best action (most visited child)
    best_action = mcts.select_action()
    print("Best action chosen by MCTS:", best_action)

    # Advance the MCTS tree to the chosen action
    mcts.advance(best_action)

if __name__ == "__main__":
    main()