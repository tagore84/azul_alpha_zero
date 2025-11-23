import sys
import os
import torch
import torch.nn as nn
import unittest

# Add src to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from net.azul_net import AzulNet
from players.deep_mcts_player import DeepMCTSPlayer

class TestArchitecture(unittest.TestCase):
    def test_azul_net_structure(self):
        """Verify AzulNet has the new hidden layer."""
        in_channels = 4
        global_size = 58
        action_size = 100
        model = AzulNet(in_channels, global_size, action_size)
        
        # Check for existence of policy_fc1
        self.assertTrue(hasattr(model, 'policy_fc1'), "AzulNet missing policy_fc1")
        self.assertIsInstance(model.policy_fc1, nn.Linear)
        
        # Check dimensions
        # Input to policy_fc1: 2*5*5 + global_size = 50 + 58 = 108
        self.assertEqual(model.policy_fc1.in_features, 108)
        self.assertEqual(model.policy_fc1.out_features, 256) # Default value_hidden
        
        # Input to policy_fc: 256
        self.assertEqual(model.policy_fc.in_features, 256)
        self.assertEqual(model.policy_fc.out_features, action_size)

    def test_forward_pass(self):
        """Verify forward pass works with new architecture."""
        in_channels = 4
        global_size = 58
        action_size = 100
        model = AzulNet(in_channels, global_size, action_size)
        
        batch_size = 2
        x_spatial = torch.randn(batch_size, in_channels, 5, 5)
        x_global = torch.randn(batch_size, global_size)
        
        pi, v = model(x_spatial, x_global)
        
        self.assertEqual(pi.shape, (batch_size, action_size))
        self.assertEqual(v.shape, (batch_size,))

    def test_deep_mcts_player_loading(self):
        """Verify DeepMCTSPlayer can load a model with the new architecture."""
        # Create a dummy checkpoint with new architecture
        in_channels = 4
        global_size = 58
        action_size = 100
        model = AzulNet(in_channels, global_size, action_size)
        
        checkpoint_path = 'test_checkpoint.pt'
        torch.save({'model_state': model.state_dict()}, checkpoint_path)
        
        try:
            # Try to load it
            player = DeepMCTSPlayer(checkpoint_path, device='cpu')
            self.assertIsInstance(player.net, AzulNet)
            # Verify inferred dimensions
            self.assertEqual(player.net.in_channels, in_channels)
            self.assertEqual(player.net.global_size, global_size)
        finally:
            if os.path.exists(checkpoint_path):
                os.remove(checkpoint_path)

if __name__ == '__main__':
    unittest.main()
