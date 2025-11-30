
import unittest
import numpy as np
from players.random_plus_player import RandomPlusPlayer

class TestRandomPlusPlayer(unittest.TestCase):
    def test_predict_minimizes_floor(self):
        player = RandomPlusPlayer()
        
        # Mock observation
        # 2 Players, 5 Factories, 5 Colors
        # Factory 0 has 3 tiles of color 0 (Blue)
        factories = np.zeros((5, 5), dtype=int)
        factories[0][0] = 3
        
        center = np.zeros(5, dtype=int)
        
        # Player 0 board
        pattern_lines = [
            np.full(1, -1, dtype=int), # Cap 1
            np.full(2, -1, dtype=int), # Cap 2
            np.full(3, -1, dtype=int), # Cap 3
            np.full(4, -1, dtype=int), # Cap 4
            np.full(5, -1, dtype=int), # Cap 5
        ]
        wall = np.full((5, 5), -1, dtype=int)
        
        players = [
            {'pattern_lines': pattern_lines, 'wall': wall, 'score': 0, 'floor_line': np.full(7, -1)},
            {'pattern_lines': pattern_lines, 'wall': wall, 'score': 0, 'floor_line': np.full(7, -1)}
        ]
        
        obs = {
            'factories': factories,
            'center': center,
            'players': players,
            'current_player': 0,
            'round': 0
        }
        
        # Expected behavior:
        # Taking 3 tiles of color 0 from Factory 0.
        # Dest 0 (Cap 1): 3 - 1 = 2 floor.
        # Dest 1 (Cap 2): 3 - 2 = 1 floor.
        # Dest 2 (Cap 3): 3 - 3 = 0 floor.
        # Dest -1 (Floor): 3 floor.
        
        # Player should choose Dest 2 (or 3 or 4 which also have 0 floor).
        
        action = player.predict(obs)
        print(f"Action chosen: {action}")
        
        self.assertIsNotNone(action)
        source, color, dest = action
        self.assertEqual(source, 0)
        self.assertEqual(color, 0)
        self.assertIn(dest, [2, 3, 4]) # All these have capacity >= 3, so 0 floor.

    def test_predict_forced_floor(self):
        player = RandomPlusPlayer()
        
        # Factory 0 has 5 tiles of color 0 (Blue) - impossible in real game but useful for test
        factories = np.zeros((5, 5), dtype=int)
        factories[0][0] = 5
        center = np.zeros(5, dtype=int)
        
        # Player 0 board - all lines full except line 0 (cap 1)
        pattern_lines = [
            np.full(1, -1, dtype=int), # Cap 1 (Empty)
            np.full(2, 1, dtype=int),  # Cap 2 (Full of color 1)
            np.full(3, 1, dtype=int),  # Cap 3 (Full of color 1)
            np.full(4, 1, dtype=int),  # Cap 4 (Full of color 1)
            np.full(5, 1, dtype=int),  # Cap 5 (Full of color 1)
        ]
        wall = np.full((5, 5), -1, dtype=int)
        
        players = [
            {'pattern_lines': pattern_lines, 'wall': wall, 'score': 0, 'floor_line': np.full(7, -1)},
            {'pattern_lines': pattern_lines, 'wall': wall, 'score': 0, 'floor_line': np.full(7, -1)}
        ]
        
        obs = {
            'factories': factories,
            'center': center,
            'players': players,
            'current_player': 0,
            'round': 0
        }
        
        # Valid moves for color 0:
        # Dest 0 (Cap 1): 5 - 1 = 4 floor.
        # Dest -1: 5 floor.
        # Other dests invalid (wrong color).
        
        # Should choose Dest 0.
        
        action = player.predict(obs)
        print(f"Action chosen: {action}")
        
        self.assertIsNotNone(action)
        source, color, dest = action
        self.assertEqual(source, 0)
        self.assertEqual(color, 0)
        self.assertEqual(dest, 0)

if __name__ == '__main__':
    unittest.main()
