#src/net/azul_net.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

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
    ):
        super(AzulNet, self).__init__()
        # Initial convolution
        self.conv_in = nn.Conv2d(in_channels, channels, kernel_size=3, padding=1)
        self.bn_in   = nn.BatchNorm2d(channels)
        # Residual blocks
        self.res_blocks = nn.Sequential(
            *[ResBlock(channels) for _ in range(num_blocks)]
        )
        # Policy head
        self.policy_conv = nn.Conv2d(channels, 2, kernel_size=1)
        self.policy_bn   = nn.BatchNorm2d(2)
        self.policy_fc   = nn.Linear(2 * 5 * 5 + global_size, action_size)
        # Value head
        self.value_conv = nn.Conv2d(channels, 1, kernel_size=1)
        self.value_bn   = nn.BatchNorm2d(1)
        self.value_fc1  = nn.Linear(1 * 5 * 5 + global_size, value_hidden)
        self.value_fc2  = nn.Linear(value_hidden, 1)
        # store sizes for predict()
        self.in_channels = in_channels
        self.global_size = global_size

    def forward(
        self,
        x_spatial: torch.Tensor,
        x_global: torch.Tensor
    ) -> tuple:
        """
        Args:
            x_spatial: (B, in_channels, 5, 5)
            x_global:  (B, global_size)
        Returns:
            pi_logits: (B, action_size)
            value:     (B,) in [-1,1]
        """
        # Input conv + res
        x = F.relu(self.bn_in(self.conv_in(x_spatial)))
        x = self.res_blocks(x)
        # Policy head
        p = F.relu(self.policy_bn(self.policy_conv(x)))
        p = p.view(p.size(0), -1)  # (B, 2*5*5)
        p = torch.cat([p, x_global], dim=1)
        pi_logits = self.policy_fc(p)
        # Value head
        v = F.relu(self.value_bn(self.value_conv(x)))
        v = v.view(v.size(0), -1)  # (B, 1*5*5)
        v = torch.cat([v, x_global], dim=1)
        v = F.relu(self.value_fc1(v))
        v = torch.tanh(self.value_fc2(v)).squeeze(-1)
        return pi_logits, v

    def predict(self, obs_batch: np.ndarray):
        """
        Predict interface for MCTS:
        - obs_batch: numpy array of shape (batch, obs_size)
          where obs_size = in_channels*5*5 + global_size
        Returns:
        - pi_logits: numpy array (batch, action_size)
        - values:    numpy array (batch,)
        """
        import torch
        self.eval()
        batch, obs_size = obs_batch.shape
        device = next(self.parameters()).device

        # split flat observation into spatial and global parts
        spatial_size = self.in_channels * 5 * 5
        spatial_flat = obs_batch[:, :spatial_size]
        x_spatial = torch.from_numpy(spatial_flat).float().to(device).view(
            batch, self.in_channels, 5, 5
        )

        global_flat = obs_batch[:, spatial_size:]
        x_global = torch.from_numpy(global_flat).float().to(device)

        with torch.no_grad():
            pi_logits, values = self.forward(x_spatial, x_global)
        return pi_logits.cpu().numpy(), values.cpu().numpy()