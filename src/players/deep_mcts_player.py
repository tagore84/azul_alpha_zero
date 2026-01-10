# File: src/players/deep_mcts_player.py

import torch
import numpy as np

from net.azul_net import AzulNet

from azul.env import AzulEnv
from mcts.mcts import MCTS

from .base_player import BasePlayer

class DeepMCTSPlayer(BasePlayer):
    def __init__(self, model_path, device='cpu', mcts_iters=300, cpuct=1.0, single_player_mode=True, temperature=0.0):
        super().__init__()
        self.device = torch.device(device)
        self.temperature = temperature
        # Load checkpoint and extract model state
        checkpoint = torch.load(model_path, map_location=self.device)
        state_dict = checkpoint.get('model_state', checkpoint)
        
        # Infer network dimensions from checkpoint
        in_channels = state_dict['conv_in.weight'].shape[1]
        
        # For Phase 2 architecture:
        # policy_fc1.weight has shape [256, spatial_flat + factories_flat + global_size]
        # value_fc1.weight has shape [256, spatial_flat + factories_flat + global_size]
        # spatial_flat = 2 * 5 * 5 = 50 (policy conv outputs 2 channels)
        # spatial_flat = 1 * 5 * 5 = 25 (value conv outputs 1 channel)
        # factories_flat = (N + 1) * embed_dim, where N=5, we need to infer embed_dim
        
        # For now, use a simpler approach: initialize env to get correct sizes
        env = AzulEnv()
        obs_flat = env.encode_observation(env.reset(initial=True))
        total_obs_size = obs_flat.shape[0]
        # Fix spatial size calculation to match train_loop_v5.py (20 channels for 2 players)
        # 2 players * (Pattern + Wall) * 5 colors = 20 channels
        spatial_size = env.num_players * 2 * 5 * 5 * 5
        factories_size = (env.N + 1) * 5
        global_size = total_obs_size - spatial_size - factories_size
        action_size = env.action_size
        
        # Build and load network
        self.net = AzulNet(
            in_channels=in_channels,
            global_size=global_size,
            action_size=action_size,
            factories_count=env.N
        )
        self.net.load_state_dict(state_dict)
        self.net.to(self.device)
        self.net.eval()
        
        # Prepare prototype environment for MCTS
        self.prototype_env = env
        # Initialize MCTS searcher
        self.mcts = MCTS(self.prototype_env,
                         self.net,
                         simulations=mcts_iters,
                         cpuct=cpuct,
                         single_player_mode=single_player_mode)

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
        action = self.mcts.select_action(temperature=self.temperature)
        return action