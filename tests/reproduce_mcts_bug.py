
import unittest
import numpy as np
from unittest.mock import MagicMock
from mcts.mcts import MCTS

class MockEnv:
    def __init__(self):
        self.current_player = 0
        self.players = [{'wall': []}, {'wall': []}] # Dummy
        self.state_name = "ROOT"

    def clone(self):
        new = MockEnv()
        new.current_player = self.current_player
        new.state_name = self.state_name
        return new

    def get_valid_actions(self):
        return [(0, 0, 0), (0, 0, 1)] # Two dummy actions

    def action_to_index(self, action):
        return action[2]

    def _get_obs(self):
        return {}

    def encode_observation(self, obs):
        return np.array([])

    def step(self, action, is_sim=False):
        # Action 0 -> State A (Good for P0, Bad for P1)
        # Action 1 -> State B (Bad for P0, Good for P1)
        if action[2] == 0:
            self.state_name = "A"
        else:
            self.state_name = "B"
        self.current_player = 1 - self.current_player
        return {}, 0, False, {}
    
    def get_winner(self):
        return []

class MockModel:
    def predict(self, obs):
        # Returns (policy_logits, value)
        # Value is for the current player
        return np.array([[0.0, 0.0]]), np.array([0.0])

class TestMCTS(unittest.TestCase):
    def test_backpropagate_logic(self):
        env = MockEnv()
        # Mock model to return specific values for states
        model = MockModel()
        
        # We need to control the value returned during expansion/simulation
        # But MCTS calls model.predict inside run/expand.
        # Let's subclass MCTS or mock the model's predict to return based on state?
        # But model receives encoded observation.
        
        # Let's just manually build the tree and call backpropagate to see what happens.
        mcts = MCTS(env, model, simulations=1)
        
        root = mcts.root # P0 to move
        
        # Create child A (Action 0)
        env_a = env.clone()
        env_a.step((0,0,0)) # Now P1 to move. State A.
        node_a = MCTS.Node(env_a, parent=root, prior=0.5)
        root.children[(0,0,0)] = node_a
        
        # Create child B (Action 1)
        env_b = env.clone()
        env_b.step((0,0,1)) # Now P1 to move. State B.
        node_b = MCTS.Node(env_b, parent=root, prior=0.5)
        root.children[(0,0,1)] = node_b
        
        # Scenario 1: Node A is evaluated.
        # P1 is to move at Node A.
        # Suppose Node A is BAD for P1 (Value = -1.0). This means it was GOOD for P0.
        # We backpropagate -1.0.
        path_a = [root, node_a]
        mcts.backpropagate(path_a, -1.0)
        
        # Scenario 2: Node B is evaluated.
        # P1 is to move at Node B.
        # Suppose Node B is GOOD for P1 (Value = +1.0). This means it was BAD for P0.
        # We backpropagate +1.0.
        path_b = [root, node_b]
        mcts.backpropagate(path_b, 1.0)
        
        # Now check values of Node A and Node B.
        # Node A should be attractive for P0 (Root).
        # Node B should be repulsive for P0 (Root).
        
        print(f"Node A (Good for P0) value: {node_a.value}")
        print(f"Node B (Bad for P0) value: {node_b.value}")
        
        # If MCTS works correctly for Zero-Sum:
        # Root (P0) chooses child with max UCB.
        # UCB uses node.value.
        # So Node A should have HIGHER value than Node B.
        
        self.assertTrue(node_a.value > node_b.value, 
                        f"Node A ({node_a.value}) should be > Node B ({node_b.value}) because A is good for P0")

if __name__ == '__main__':
    unittest.main()
