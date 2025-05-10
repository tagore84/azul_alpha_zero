import sys
import os
import unittest
import numpy as np

# Ensure src folder is on PYTHONPATH for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from azul.env import AzulEnv
from azul.rules import Color

class TestEnv(unittest.TestCase):

    def setUp(self):
        # seed random for reproducibility
        np.random.seed(0)
        self.env = AzulEnv(num_players=2, factories_count=5)

    def test_reset(self):
        obs = self.env.reset()
        # Bag sum = initial 100 - (num_factories*4) = 100 - 20 = 80
        self.assertEqual(int(obs['bag'].sum()), 100 - self.env.N * 4)
        # Factories sum = num_factories * 4
        self.assertEqual(int(obs['factories'].sum()), self.env.N * 4)
        # Center is empty
        self.assertTrue((obs['center'] == 0).all())
        # First player token present
        self.assertTrue(obs['first_player_token'])
        # Current player is 0
        self.assertEqual(obs['current_player'], 0)
        # Players initial state
        for p in obs['players']:
            # pattern lines correct length and all -1
            for i, line in enumerate(p['pattern_lines']):
                self.assertEqual(len(line), i + 1)
                self.assertTrue((line == -1).all())
            # wall all -1
            self.assertTrue((p['wall'] == -1).all())
            # floor line zeros
            self.assertTrue((p['floor_line'] == 0).all())
            # score zero
            self.assertEqual(p['score'], 0)

    def test_step_take_from_factory(self):
        # override factories and center to deterministic state
        self.env.factories[:] = 0
        self.env.center[:] = 0
        # place 3 tiles of color index 2 in factory 0
        self.env.factories[0, 2] = 3
        # ensure current player is 0
        self.env.current_player = 0
        obs, reward, done, info = self.env.step((0, 2, 0))
        # Factory emptied for that color
        self.assertEqual(int(obs['factories'][0, 2]), 0)
        # Center received moved tiles (3 of color 2)
        self.assertEqual(int(obs['center'][2]), 3)
        # Pattern line 0 got one tile (capacity 1)
        self.assertEqual(list(obs['players'][0]['pattern_lines'][0]), [2])
        # Floor line got two overflow tiles (color+1 = 3)
        floor = obs['players'][0]['floor_line']
        self.assertEqual(int(floor[0]), 3)
        self.assertEqual(int(floor[1]), 3)
        # Reward zero, not done, next player is 1
        self.assertEqual(reward, 0)
        self.assertFalse(done)
        self.assertEqual(obs['current_player'], 1)

if __name__ == '__main__':
    unittest.main()
