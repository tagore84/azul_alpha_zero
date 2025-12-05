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

    def train_epoch(self, train_loader: torch.utils.data.DataLoader, epoch: int, max_grad_norm: float = 1.0):
        """
        Perform one training epoch.
        Logs training loss to TensorBoard if enabled.
        """
        self.model.train()
        total_loss = 0.0
        loss_pi = 0.0
        loss_v = 0.0
        
        for batch_idx, batch in enumerate(train_loader):
            # Unpack batch: obs_spatial, obs_factories, obs_global, target_pi, target_v
            obs_spatial   = batch['spatial'].to(self.device)
            obs_factories = batch['factories'].to(self.device)
            obs_global    = batch['global'].to(self.device)
            target_pi     = batch['pi'].to(self.device)
            target_v      = batch['v'].to(self.device).float()

            # Forward pass
            # We don't have action_mask in the dataset yet, so we pass None (model will use ones)
            pi_logits, value = self.model(obs_spatial, obs_global, obs_factories)

            # Compute losses
            # Policy loss: Cross-entropy between MCTS policy (target) and network policy
            # Using cross-entropy with soft targets (more stable than KL divergence)
            log_pi = torch.nn.functional.log_softmax(pi_logits, dim=1)
            l_pi = -(target_pi * log_pi).sum(dim=1).mean()  # Cross-entropy with soft targets
            l_v  = torch.nn.functional.mse_loss(value, target_v)
            loss = l_pi + l_v

            # NaN Detection: Skip batch if NaN detected to prevent model corruption
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"[trainer] WARNING: NaN/Inf loss detected in batch {batch_idx}, skipping...", flush=True)
                continue

            # Backward and optimize
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient Clipping
            if max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)
                
            self.optimizer.step()

            total_loss += loss.item()
            loss_pi += l_pi.item() # Accumulate for logging
            loss_v += l_v.item()

            if self.writer:
                global_step = epoch * len(train_loader) + batch_idx
                self.writer.add_scalar('train/loss', loss.item(), global_step)
                self.writer.add_scalar('train/loss_policy', l_pi.item(), global_step)
                self.writer.add_scalar('train/loss_value', l_v.item(), global_step)
                if l_v.item() > 0:
                    self.writer.add_scalar('train/loss_ratio', l_pi.item() / l_v.item(), global_step)

        avg_loss = total_loss / len(train_loader)
        avg_pi = loss_pi / len(train_loader)
        avg_v = loss_v / len(train_loader)
        return {'total': avg_loss, 'policy': avg_pi, 'value': avg_v}

    def evaluate(self, val_loader: torch.utils.data.DataLoader, epoch: int):
        """
        Evaluate model on validation set.
        Logs validation loss to TensorBoard if enabled.
        """
        self.model.eval()
        total_loss = 0.0
        with torch.no_grad():
            for batch_idx, batch in enumerate(val_loader):
                obs_spatial   = batch['spatial'].to(self.device)
                obs_factories = batch['factories'].to(self.device)
                obs_global    = batch['global'].to(self.device)
                target_pi     = batch['pi'].to(self.device)
                target_v      = batch['v'].to(self.device)

                pi_logits, value = self.model(obs_spatial, obs_global, obs_factories)
                log_pi = torch.nn.functional.log_softmax(pi_logits, dim=1)
                loss_pi = -(target_pi * log_pi).sum(dim=1).mean()  # Cross-entropy with soft targets
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
            epochs: int = 10, checkpoint_dir: str = None,
            max_grad_norm: float = 1.0):
        """
        Run the full training loop.
        Saves checkpoints to checkpoint_dir if provided.
        Returns a dictionary with training history.
        """
        history = {'train_loss': [], 'train_loss_policy': [], 'train_loss_value': [], 'val_loss': []}
        
        # Scheduler: Cosine Annealing within the cycle
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=epochs, eta_min=1e-6)
        
        for epoch in range(1, epochs + 1):
            # Pass max_grad_norm to train_epoch (need to update train_epoch signature or handle here)
            # Actually, let's update train_epoch to handle clipping
            epoch_losses = self.train_epoch(train_loader, epoch, max_grad_norm)
            
            # Step scheduler
            scheduler.step()
            current_lr = scheduler.get_last_lr()[0]
            
            print(f"[trainer] Epoch {epoch}/{epochs} - Loss: {epoch_losses['total']:.4f} (π: {epoch_losses['policy']:.4f}, v: {epoch_losses['value']:.4f}) - LR: {current_lr:.2e}")
            
            if self.writer:
                global_step = epoch * len(train_loader)
                self.writer.add_scalar('train/lr', current_lr, global_step)
            
            history['train_loss'].append(epoch_losses['total'])
            history['train_loss_policy'].append(epoch_losses['policy'])
            history['train_loss_value'].append(epoch_losses['value'])

            if val_loader:
                val_loss = self.evaluate(val_loader, epoch)
                print(f"[trainer] Epoch {epoch}/{epochs} - Val   Loss: {val_loss:.4f}")
                history['val_loss'].append(val_loss)
            else:
                history['val_loss'].append(None)

            if checkpoint_dir:
                os.makedirs(checkpoint_dir, exist_ok=True)
                checkpoint_path = os.path.join(checkpoint_dir, f"model_epoch_{epoch:03}.pt")
                torch.save({'model_state': self.model.state_dict()}, checkpoint_path)
        
        return history