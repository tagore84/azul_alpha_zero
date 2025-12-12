import sys
import os
import unittest
import numpy as np

# Add project src folder to PYTHONPATH
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from azul.env import AzulEnv

class TestMaxRounds(unittest.TestCase):
    def test_max_rounds_enforcement(self):
        """Test that the game ends when max_rounds is reached."""
        max_rounds = 2
        env = AzulEnv(max_rounds=max_rounds)
        env.reset()
        
        # Verify init
        self.assertEqual(env.max_rounds, max_rounds)
        
        # Round 1
        self.assertEqual(env.round_count, 1)
        # Simulate round 1 end by forcing all tiles to be taken
        # We can cheat by clearing factories and center
        env.factories[:] = 0
        env.center[:] = 0
        
        # This checks is_round_over -> True
        self.assertTrue(env._is_round_over())
        
        # Step call usually handles end round logic at the END of a move
        # But here we want to test _end_round logic specifically.
        
        # Let's verify normal transition to round 2
        game_over = env._end_round()
        self.assertFalse(game_over, "Game should not end after round 1")
        self.assertEqual(env.round_count, 2)
        self.assertEqual(env.termination_reason, "normal_end")
        
        # Round 2 (Last round)
        # Simulate round 2 end
        env.factories[:] = 0
        env.center[:] = 0
        
        game_over = env._end_round()
        self.assertTrue(game_over, "Game MUST end after round 2 (max_rounds=2)")
        self.assertEqual(env.termination_reason, "max_rounds_reached")
        
        print("\nTest passed: Game ended correctly at max_rounds with reason 'max_rounds_reached'")

if __name__ == '__main__':
    unittest.main()
