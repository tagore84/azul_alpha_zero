#src/net/azul_net.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from azul.env import AzulEnv
from mcts.mcts import MCTS
import copy

class ResBlock(nn.Module):
    def __init__(self, channels: int):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1   = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2   = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + residual)

class AzulNet(nn.Module):
    """
    Conv-Residual network for Azul Zero.
    Inputs:
      - spatial input: tensor (batch, in_channels, 5, 5)
      - global input:  tensor (batch, global_size)
    Outputs:
      - pi_logits: tensor (batch, action_size)
      - value:     tensor (batch,) in [-1,1]
    """
    def __init__(
        self,
        in_channels: int,
        global_size: int,
        action_size: int,
        channels: int = 64,
        num_blocks: int = 4,
        value_hidden: int = 256,
        factories_count: int = 5,
        factory_embed_dim: int = 32,
    ):
        super(AzulNet, self).__init__()
        self.factories_count = factories_count
        self.factory_embed_dim = factory_embed_dim
        
        # Initial convolution
        self.conv_in = nn.Conv2d(in_channels, channels, kernel_size=3, padding=1)
        self.bn_in   = nn.BatchNorm2d(channels)
        # Residual blocks
        self.res_blocks = nn.Sequential(
            *[ResBlock(channels) for _ in range(num_blocks)]
        )
        
        # Factory Transformer
        # Input: (Batch, N+1, 5) -> Embed -> (Batch, N+1, 32)
        self.factory_embedding = nn.Linear(5, factory_embed_dim)
        # Learnable positional encoding to distinguish Center from Factories
        self.factory_pos_embedding = nn.Parameter(torch.randn(1, factories_count + 1, factory_embed_dim) * 0.02)
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=factory_embed_dim, nhead=4, dim_feedforward=64, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        
        # Flattened factory features size: (N + 1) * embed_dim
        self.factory_out_size = (factories_count + 1) * factory_embed_dim
        
        # Shared Trunk (Fusion Layer)
        combined_size = 3 * 5 * 5 + global_size + self.factory_out_size
        self.layer_norm = nn.LayerNorm(combined_size)
        self.fusion_fc1 = nn.Linear(combined_size, value_hidden)
        self.fusion_fc2 = nn.Linear(value_hidden, value_hidden)
        
        # Policy head
        self.policy_conv = nn.Conv2d(channels, 2, kernel_size=1)
        self.policy_bn   = nn.BatchNorm2d(2)
        # Input: shared trunk output (no action_mask concatenation)
        self.policy_fc1  = nn.Linear(value_hidden, value_hidden)
        self.policy_fc   = nn.Linear(value_hidden, action_size)
        
        # Value head
        self.value_conv = nn.Conv2d(channels, 1, kernel_size=1)
        self.value_bn   = nn.BatchNorm2d(1)
        self.value_fc1  = nn.Linear(value_hidden, value_hidden)
        self.value_fc2  = nn.Linear(value_hidden, 1) # Linear output (no Tanh)
        
        # store sizes for predict()
        self.in_channels = in_channels
        self.global_size = global_size
        self.action_size = action_size

    def forward(
        self,
        x_spatial: torch.Tensor,
        x_global: torch.Tensor,
        x_factories: torch.Tensor,
        action_mask: torch.Tensor = None
    ) -> tuple:
        """
        Args:
            x_spatial: (B, in_channels, 5, 5) - Typically 20 channels (2 players * 2 grids * 5 colors)
            x_global:  (B, global_size) - excluding factories
            x_factories: (B, N+1, 5) - factories + center counts
            action_mask: (B, action_size) - binary mask (1=legal, 0=illegal), optional
        Returns:
            pi_logits: (B, action_size)
            value:     (B,) unbounded (score difference)
        """
        batch_size = x_spatial.size(0)
        
        # Input conv + res
        x = F.relu(self.bn_in(self.conv_in(x_spatial)))
        x = self.res_blocks(x)
        
        # Process factories with Transformer
        # x_factories: (B, N+1, 5)
        f = F.relu(self.factory_embedding(x_factories)) # (B, N+1, 32)
        f = f + self.factory_pos_embedding # Add position info (broadcasting over batch)
        f = self.transformer(f) # (B, N+1, 32)
        f = f.reshape(f.size(0), -1) # (B, (N+1)*32)
        
        # Combine spatial + global + factories
        p = F.relu(self.policy_bn(self.policy_conv(x)))
        p = p.view(p.size(0), -1)  # (B, 2*5*5)
        v = F.relu(self.value_bn(self.value_conv(x)))
        v = v.view(v.size(0), -1)  # (B, 1*5*5)
        
        combined = torch.cat([p, v, x_global, f], dim=1)
        
        # Normalize combined features
        combined = self.layer_norm(combined)
        
        # Shared Trunk (Fusion Layer)
        shared = F.relu(self.fusion_fc1(combined))
        shared = F.relu(self.fusion_fc2(shared))
        
        # Policy head
        p = F.relu(self.policy_fc1(shared))
        pi_logits = self.policy_fc(p)
        
        # Additive Action Masking
        if action_mask is not None:
            # mask is 1 for legal, 0 for illegal
            # Use masked_fill for stability (push illegal logits to -inf equivalent)
            pi_logits = pi_logits.masked_fill(action_mask == 0, -1e9)
        
        # Value head
        v = F.relu(self.value_fc1(shared))
        value = torch.tanh(self.value_fc2(v)).squeeze(-1) # Tanh output [-1, 1]
        
        return pi_logits, value

    def predict(self, obs_batch: np.ndarray, action_mask: np.ndarray = None):
        """
        Predict interface for MCTS:
        - obs_batch: numpy array of shape (batch, total_flat_size)
          Layout: [Spatial (C*5*5) | Factories ((N+1)*5) | Global (Rest)]
        - action_mask: optional numpy array of shape (batch, action_size) - binary mask (1=legal, 0=illegal)
        """
        self.eval()
        batch, _ = obs_batch.shape
        device = next(self.parameters()).device

        # 1. Spatial
        spatial_size = self.in_channels * 5 * 5
        spatial_flat = obs_batch[:, :spatial_size]
        x_spatial = torch.from_numpy(spatial_flat).float().to(device).view(
            batch, self.in_channels, 5, 5
        )
        
        # 2. Factories
        # factories_count + 1 (center) * 5 colors
        factories_flat_size = (self.factories_count + 1) * 5
        factories_flat = obs_batch[:, spatial_size : spatial_size + factories_flat_size]
        x_factories = torch.from_numpy(factories_flat).float().to(device).view(
            batch, self.factories_count + 1, 5
        )
        
        # 3. Global (Rest)
        global_flat = obs_batch[:, spatial_size + factories_flat_size:]
        x_global = torch.from_numpy(global_flat).float().to(device)
        
        # 4. Action mask (optional)
        if action_mask is not None:
            action_mask_tensor = torch.from_numpy(action_mask).float().to(device)
        else:
            action_mask_tensor = None

        with torch.no_grad():
            pi_logits, values = self.forward(x_spatial, x_global, x_factories, action_mask_tensor)
        return pi_logits.cpu().numpy(), values.cpu().numpy()
    
def evaluate_against_previous(current_model, previous_model, env_args, simulations, cpuct, n_games):
    """
    Play n_games between current_model (player 0) and previous_model (player 1).
    Returns wins_current, wins_previous.
    """
    wins_current = 0
    wins_prev    = 0
    for _ in range(n_games):
        env = AzulEnv(**env_args)
        obs = env.reset()
        done = False
        while not done:
            current = env.current_player  # 0 or 1
            model = current_model if current == 0 else previous_model
            if hasattr(model, "predict_without_mcts"):
                action = env.index_to_action(model.predict_without_mcts(obs))
            else:
                mcts = MCTS(
                    env.__class__(num_players=env.num_players, factories_count=env.N, max_rounds=env.max_rounds),
                    model, simulations=simulations, cpuct=cpuct
                )
                mcts.root.env.__dict__ = copy.copy(env.__dict__)
                mcts.run()
                action = mcts.select_action()
            obs, reward, done, info = env.step(action)
        
        winners = env.get_winner()
        print(f"[azul-net] Game result: {winners}, info: {info}")
        if 0 in env.get_winner():
            wins_current += 1
        elif 1 in env.get_winner():
            wins_prev += 1
    return wins_current, wins_prev