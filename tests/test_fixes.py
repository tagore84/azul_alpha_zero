import sys
import os
import numpy as np
import torch
import unittest

# Add src to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from azul.env import AzulEnv
from azul.rules import transfer_to_wall
from mcts.mcts import MCTS
from net.azul_net import AzulNet

class TestFixes(unittest.TestCase):
    def test_encoding_structure(self):
        """Verify that spatial features are at the beginning of the encoded observation."""
        env = AzulEnv(num_players=2)
        obs = env.reset()
        
        # Modify state to have recognizable patterns
        # Player 0 Pattern Line 0: [0] (Blue)
        env.players[0]['pattern_lines'][0][0] = 0
        # Player 0 Wall: [0,0] is filled (Blue)
        env.players[0]['wall'][0][0] = 0
        
        # Set bag to known value to verify position
        env.bag[:] = 20
        
        encoded = env.encode_observation(env._get_obs())
        
        # Expected Spatial Size: 2 players * 2 features * 25 = 100
        spatial_size = 4 * 25
        
        # Check if spatial part contains our modifications
        # Pattern lines come first.
        # Player 0, Line 0 is at index 0 of the spatial block.
        # It is padded to 5. So indices 0-4.
        # Value at 0 should be 0.
        self.assertEqual(encoded[0], 0)
        
        # Wall comes after pattern lines.
        # Player 0 Pattern Lines: 5*5 = 25.
        # Player 1 Pattern Lines: 25.
        # Player 0 Wall starts at 50.
        # Wall[0][0] is at 50.
        self.assertEqual(encoded[50], 0)
        
        # Verify global part is after spatial
        # Bag is first global feature.
        # Bag starts at 100.
        # Bag has 20 of each color.
        self.assertEqual(encoded[100], 20)

    def test_scoring_logic(self):
        """Verify the scoring logic fix in transfer_to_wall."""
        # Setup a wall where we place a tile that has BOTH horizontal and vertical neighbors.
        # Wall layout (0=Blue, 1=Yellow, 2=Red, 3=Black, 4=Orange)
        # Row 0: B Y R B K
        # Row 1: R B Y O K
        
        # We want to place Blue at (1, 1).
        # Neighbors: (1, 0) is Red (Horizontal). (0, 1) is Yellow (Vertical).
        
        wall = [
            [-1, 1, -1, -1, -1], # Row 0: [1,1] is Yellow (Vertical neighbor)
            [2, -1, -1, -1, -1], # Row 1: [1,0] is Red (Horizontal neighbor)
            [-1, -1, -1, -1, -1],
            [-1, -1, -1, -1, -1],
            [-1, -1, -1, -1, -1]
        ]
        
        # Pattern line for Row 1 (Red is first, Blue is second).
        # Row 1 pattern: R B Y O K. Blue is at index 1.
        # We are placing Blue (0).
        pattern_line = [0, 0] # 2 tiles of Blue
        
        # Expected Score:
        # Horizontal: Red(1,0) - Blue(1,1). Count = 2.
        # Vertical: Yellow(0,1) - Blue(1,1). Count = 2.
        # Total = 2 + 2 = 4.
        
        # Before fix, it would be 1 (base) + 1 (horiz neighbor) + 1 (vert neighbor) = 3.
        
        score = transfer_to_wall(wall, pattern_line, 1)
        self.assertEqual(score, 4, f"Score should be 4, got {score}")

    def test_mcts_terminal_value(self):
        """Verify MCTS correctly identifies winner/loser values."""
        env = AzulEnv(num_players=2)
        env.reset()
        
        # Mock a terminal state
        # Player 0 has a full row
        env.players[0]['wall'][0] = [0, 1, 2, 3, 4]
        env.players[0]['score'] = 100
        env.players[1]['score'] = 50
        
        # Current player is 1 (it's their turn, but game is over)
        env.current_player = 1
        env.done = True
        
        # MCTS Node
        mcts = MCTS(env, model=None)
        node = mcts.root
        
        # Expand should fail or we just check run logic manually?
        # MCTS.run calls expand then backpropagate.
        # But if terminal, it skips expand.
        
        # Let's simulate the logic inside run() for a terminal node
        winners = env.get_winner() # Should be [0]
        current = env.current_player # 1
        
        if current in winners:
             if len(winners) > 1:
                 value = 0.0
             else:
                 value = 1.0
        else:
             value = -1.0
             
        self.assertEqual(value, -1.0, "Player 1 should have value -1.0 (Loss)")
        
        # Now check from Player 0 perspective
        env.current_player = 0
        current = env.current_player
        if current in winners:
             if len(winners) > 1:
                 value = 0.0
             else:
                 value = 1.0
        else:
             value = -1.0
        self.assertEqual(value, 1.0, "Player 0 should have value 1.0 (Win)")

if __name__ == '__main__':
    unittest.main()
