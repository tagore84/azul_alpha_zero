
import unittest
import numpy as np
from azul.env import AzulEnv

class TestCanonicalObs(unittest.TestCase):
    def test_canonical_rotation(self):
        env = AzulEnv(num_players=2)
        env.reset()
        
        # Setup distinct states for P0 and P1
        # P0 has score 10, P1 has score 20
        env.players[0]['score'] = 10
        env.players[1]['score'] = 20
        
        # P0 has a tile in pattern line 0
        env.players[0]['pattern_lines'][0][0] = 1
        # P1 has a tile in pattern line 1
        env.players[1]['pattern_lines'][1][0] = 2
        
        # Case 1: Current player is 0
        env.current_player = 0
        obs0 = env._get_obs()
        enc0 = env.encode_observation(obs0)
        
        # Case 2: Current player is 1
        env.current_player = 1
        obs1 = env._get_obs()
        enc1 = env.encode_observation(obs1)
        
        # We need to verify that enc1 "looks like" enc0 if we were to swap the players in the state.
        # But easier: 
        # In enc0, the first part of spatial/scores should correspond to P0 (Score 10).
        # In enc1, the first part of spatial/scores should correspond to P1 (Score 20).
        
        # Let's inspect the scores part of the encoding.
        # Global parts start after spatial parts.
        # Spatial size = 2 * 5 * 5 = 50 integers per player. Total 100 integers.
        # Global parts:
        # Bag (5) + Discard (5) + Factories (6*5=30) + Center (5) + FirstToken (1) = 46
        # Floor lines (2*7=14)
        # Scores (2)
        
        # Total size before scores: 100 (spatial) + 46 (global misc) + 14 (floors) = 160
        # So scores are at index 160 and 161.
        
        # NOTE: The exact indices depend on the implementation details (flattening order).
        # Let's verify by checking the values directly.
        
        # In enc0: P0 is first. Score at 160 should be 10. Score at 161 should be 20.
        score_p0_enc0 = enc0[-2] # Assuming scores are at the end (before removed current_player)
        score_p1_enc0 = enc0[-1]
        
        # In enc1: P1 is first. Score at 160 should be 20. Score at 161 should be 10.
        score_p0_enc1 = enc1[-2]
        score_p1_enc1 = enc1[-1]
        
        print(f"Enc0 scores: {score_p0_enc0}, {score_p1_enc0}")
        print(f"Enc1 scores: {score_p0_enc1}, {score_p1_enc1}")
        
        self.assertEqual(score_p0_enc0, 10)
        self.assertEqual(score_p1_enc0, 20)
        
        self.assertEqual(score_p0_enc1, 20)
        self.assertEqual(score_p1_enc1, 10)
        
        # Verify spatial parts
        # P0 has 1 at pattern line 0. P1 has 2 at pattern line 1.
        # In enc0: First 25 values are P0 pattern lines.
        # In enc1: First 25 values are P1 pattern lines.
        
        # P0 pattern line 0 is at index 0.
        self.assertEqual(enc0[0], 1)
        
        # P1 pattern line 1 is at index 5 (row 1, col 0) + offset.
        # In enc1, P1 is first, so it should be at index 5.
        self.assertEqual(enc1[5], 2)

if __name__ == '__main__':
    unittest.main()
