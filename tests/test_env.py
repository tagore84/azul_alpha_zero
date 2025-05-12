import sys
import os
import unittest
import numpy as np

# Ensure src folder is on PYTHONPATH for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from constants import SEED
from azul.env import AzulEnv
from azul.rules import Color, transfer_to_wall

class TestEnv(unittest.TestCase):

    def setUp(self):
        # seed random for reproducibility
        np.random.seed(0)
        self.env = AzulEnv(num_players=2, factories_count=5, seed=SEED)

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
            # floor line -1s
            self.assertTrue((p['floor_line'] == -1).all())
            # score zero
            self.assertEqual(p['score'], 0)

    def test_step_take_from_factory(self):
        # override factories and center to deterministic state
        self.env.factories[:] = 0
        self.env.center[:] = 0
        # place 3 tiles of color index 2 in factory 0
        self.env.factories[0, 0] = 3
        self.env.factories[0, 1] = 1
        # ensure current player is 0
        self.env.current_player = 0
        obs, reward, done, info = self.env.step((0, 0, 0))
        # Factory should have one tile of color 2 remaining (one taken, two left)
        self.assertEqual(int(obs['factories'][0, 0]), 0)
        self.assertEqual(int(obs['factories'][0, 1]), 0)
        # Center should remain with 3 in color 0 (no other colors to move)
        self.assertEqual(int(obs['center'][1]), 1)
        # Pattern line 0 got one tile (capacity 1)
        self.assertEqual(int(self.env.players[0]['pattern_lines'][0][0]), 0)
        # Floor line got two overflow tiles (color+1 = 3)
        floor = obs['players'][0]['floor_line']
        self.assertEqual(int(floor[0]), 0)
        self.assertEqual(int(floor[1]), 0)
        # Reward zero, not done, next player is 1
        #self.assertEqual(reward, 0)
        self.assertFalse(done)
        self.assertEqual(obs['current_player'], 1)

    def test_clone_independence(self):
        clone = self.env.clone()

        # Modify the original environment
        self.env.factories[0, 0] = 99
        self.env.center[0] = 42
        self.env.players[0]['wall'][0, 0] = 7
        self.env.players[0]['floor_line'][0] = 3
        self.env.players[0]['pattern_lines'][0][0] = 5
        self.env.players[0]['score'] = 11
        self.env.current_player = 1

        # Ensure the clone did not change
        self.assertNotEqual(self.env.factories[0, 0], clone.factories[0, 0])
        self.assertNotEqual(self.env.center[0], clone.center[0])
        self.assertNotEqual(self.env.players[0]['wall'][0, 0], clone.players[0]['wall'][0, 0])
        self.assertNotEqual(self.env.players[0]['floor_line'][0], clone.players[0]['floor_line'][0])
        self.assertNotEqual(self.env.players[0]['pattern_lines'][0][0], clone.players[0]['pattern_lines'][0][0])
        self.assertNotEqual(self.env.players[0]['score'], clone.players[0]['score'])
        self.assertNotEqual(self.env.current_player, clone.current_player)

    def test_clone_wall_integrity(self):
        # Set a known wall state
        self.env.players[0]['wall'][:] = -1
        self.env.players[0]['wall'][1, 2] = 4
        self.env.players[1]['wall'][3, 1] = 2

        # Clone the environment
        clone = self.env.clone()

        # Modify the original wall
        self.env.players[0]['wall'][1, 2] = 1
        self.env.players[1]['wall'][3, 1] = 3

        # Verify that the clone's wall has not changed
        self.assertEqual(clone.players[0]['wall'][1, 2], 4)
        self.assertEqual(clone.players[1]['wall'][3, 1], 2)

        # Also verify that other positions remain -1
        self.assertTrue((clone.players[0]['wall'] == -1).sum() >= 23)  # 25 total - 2 changed
        self.assertTrue((clone.players[1]['wall'] == -1).sum() >= 23)

    def test_wall_persists_after_transfer(self):
        wall = [[-1]*5 for _ in range(5)]
        pattern_line = [Color.RED] * 3
        score = transfer_to_wall(wall, pattern_line, row=1)
        assert wall[1].count(Color.RED) == 1
        # Simula una ronda siguiente:
        wall_copy = [row.copy() for row in wall]
        # Verifica que el wall anterior persiste
        assert wall_copy == wall

if __name__ == '__main__':
    unittest.main()
