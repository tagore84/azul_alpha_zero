import os
import torch
from torch.utils.tensorboard import SummaryWriter
from train_azul import train

def main():
    verbose = False
    n_games = 100
    simulations = 200
    cpuct = 1.0
    batch_size = 64
    epochs = 20
    lr = 1e-3
    train_ratio = 0.9
    log_dir = 'logs'
    checkpoint_dir = 'data/checkpoint_dir'
    resume = None
    eval_interval = 10
    eval_games = 20

    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)

    writer = SummaryWriter(log_dir=log_dir)

    train(
        n_games=n_games,
        simulations=simulations,
        cpuct=cpuct,
        batch_size=batch_size,
        epochs=epochs,
        lr=lr,
        train_ratio=train_ratio,
        log_dir=log_dir,
        checkpoint_dir=checkpoint_dir,
        resume=resume,
        eval_interval=eval_interval,
        eval_games=eval_games,
        verbose=verbose,
        writer=writer
    )

if __name__ == "__main__":
    main()
