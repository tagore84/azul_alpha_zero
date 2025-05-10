

import os
import torch
from torch.utils.tensorboard import SummaryWriter

class Trainer:
    """
    Trainer for Azul Zero network.
    Handles the training loop, logging, and checkpointing.
    """
    def __init__(self, model: torch.nn.Module, optimizer: torch.optim.Optimizer,
                 device: torch.device, log_dir: str = None):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.device = device
        # Set up TensorBoard writer if log_dir is provided
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
            self.writer = SummaryWriter(log_dir)
        else:
            self.writer = None

    def train_epoch(self, train_loader: torch.utils.data.DataLoader, epoch: int):
        """
        Perform one training epoch.
        Logs training loss to TensorBoard if enabled.
        """
        self.model.train()
        total_loss = 0.0
        for batch_idx, batch in enumerate(train_loader):
            # Unpack batch: obs_spatial, obs_global, target_pi, target_v
            obs_spatial = batch['spatial'].to(self.device)
            obs_global  = batch['global'].to(self.device)
            target_pi   = batch['pi'].to(self.device)
            target_v    = batch['v'].to(self.device)

            # Forward pass
            pi_logits, value = self.model(obs_spatial, obs_global)

            # Compute losses
            loss_pi = torch.nn.functional.cross_entropy(pi_logits, target_pi)
            loss_v  = torch.nn.functional.mse_loss(value, target_v)
            loss = loss_pi + loss_v

            # Backward and optimize
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            if self.writer:
                global_step = epoch * len(train_loader) + batch_idx
                self.writer.add_scalar('train/loss', loss.item(), global_step)

        avg_loss = total_loss / len(train_loader)
        return avg_loss

    def evaluate(self, val_loader: torch.utils.data.DataLoader, epoch: int):
        """
        Evaluate model on validation set.
        Logs validation loss to TensorBoard if enabled.
        """
        self.model.eval()
        total_loss = 0.0
        with torch.no_grad():
            for batch_idx, batch in enumerate(val_loader):
                obs_spatial = batch['spatial'].to(self.device)
                obs_global  = batch['global'].to(self.device)
                target_pi   = batch['pi'].to(self.device)
                target_v    = batch['v'].to(self.device)

                pi_logits, value = self.model(obs_spatial, obs_global)
                loss_pi = torch.nn.functional.cross_entropy(pi_logits, target_pi)
                loss_v  = torch.nn.functional.mse_loss(value, target_v)
                loss = loss_pi + loss_v

                total_loss += loss.item()
                if self.writer:
                    global_step = epoch * len(val_loader) + batch_idx
                    self.writer.add_scalar('val/loss', loss.item(), global_step)

        avg_loss = total_loss / len(val_loader)
        return avg_loss

    def fit(self, train_loader: torch.utils.data.DataLoader,
            val_loader: torch.utils.data.DataLoader = None,
            epochs: int = 10, checkpoint_dir: str = None):
        """
        Run the full training loop.
        Saves checkpoints to checkpoint_dir if provided.
        """
        for epoch in range(1, epochs + 1):
            train_loss = self.train_epoch(train_loader, epoch)
            print(f"Epoch {epoch}/{epochs} - Train Loss: {train_loss:.4f}")

            if val_loader:
                val_loss = self.evaluate(val_loader, epoch)
                print(f"Epoch {epoch}/{epochs} - Val   Loss: {val_loss:.4f}")

            # Save checkpoint
            if checkpoint_dir:
                os.makedirs(checkpoint_dir, exist_ok=True)
                ckpt_path = os.path.join(checkpoint_dir, f"model_epoch_{epoch}.pth")
                torch.save(self.model.state_dict(), ckpt_path)
                print(f"Saved checkpoint: {ckpt_path}")