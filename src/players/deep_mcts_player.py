# File: src/players/deep_mcts_player.py

import torch
import numpy as np

from net.azul_net import AzulNet

from azul.env import AzulEnv
from mcts.mcts import MCTS

from .base_player import BasePlayer

class DeepMCTSPlayer(BasePlayer):
    def __init__(self, model_path, device='cpu', mcts_iters=200, cpuct=1.0):
        super().__init__()
        self.device = torch.device(device)
        # Load checkpoint and extract model state
        checkpoint = torch.load(model_path, map_location=self.device)
        state_dict = checkpoint.get('model_state', checkpoint)
        # Infer network dimensions from checkpoint
        in_channels = state_dict['conv_in.weight'].shape[1]
        # Derive policy head sizes
        policy_fc_weight = state_dict['policy_fc.weight']
        spatial_dim = 2 * 5 * 5  # output of policy conv head
        global_size = policy_fc_weight.shape[1] - spatial_dim
        action_size = policy_fc_weight.shape[0]
        # Build and load network
        self.net = AzulNet(in_channels, global_size, action_size)
        self.net.load_state_dict(state_dict)
        self.net.to(self.device)
        self.net.eval()
        # Prepare prototype environment for MCTS
        self.prototype_env = AzulEnv()
        # Initialize MCTS searcher
        self.mcts = MCTS(self.prototype_env,
                         self.net,
                         simulations=mcts_iters,
                         cpuct=cpuct)

    def _obs_to_env(self, obs: dict):
        """
        Copy the observation dict into the prototype_env state.
        """
        env = self.prototype_env
        env.bag = obs['bag'].copy()
        env.discard = obs['discard'].copy()
        env.factories = obs['factories'].copy()
        env.center = obs['center'].copy()
        env.first_player_token = bool(obs['first_player_token'])
        env.current_player = int(obs['current_player'])
        env.round_count = int(obs.get('round_count', env.round_count))
        for i, p_obs in enumerate(obs['players']):
            env.players[i]['pattern_lines'] = [line.copy() for line in p_obs['pattern_lines']]
            env.players[i]['wall'] = p_obs['wall'].copy()
            env.players[i]['floor_line'] = p_obs['floor_line'].copy()
            env.players[i]['score'] = int(p_obs['score'])

    def predict(self, obs: dict):
        """
        Ejecuta MCTS en el estado dado y devuelve la acción seleccionada.
        """
        # Load current observation into the prototype environment
        self._obs_to_env(obs)
        # Run MCTS from this state
        self.mcts.run(self.prototype_env)
        # Select and return an action tuple
        action = self.mcts.select_action()
        return action