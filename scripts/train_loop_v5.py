import sys
import os
import argparse
import torch
import time
from datetime import datetime

# Add project src folder to PYTHONPATH
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from azul.env import AzulEnv
from net.azul_net import AzulNet
from train.self_play import generate_self_play_games
from train.dataset import AzulDataset
from train.trainer import Trainer
from constants import SEED
from players.random_player import RandomPlayer
from players.random_plus_player import RandomPlusPlayer
from players.heuristic_player import HeuristicPlayer
from players.heuristic_min_max_mcts_player import HeuristicMinMaxMCTSPlayer
from mcts.mcts import MCTS
import copy

class TrainingLogger:
    def __init__(self, log_dir):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        self.log_file = os.path.join(log_dir, "training.log")
        self.monitor_file = os.path.join(log_dir, "training_monitor.log")
        self.buffer = []
        
        # Initialize log file
        with open(self.log_file, "a") as f:
            f.write(f"\n{'='*20}\n[{datetime.now()}] Training Session Started (V5 - Scaling)\n{'='*20}\n")

    def log(self, msg):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        formatted_msg = f"[{timestamp}] {msg}"
        print(formatted_msg) # Print to console
        self.buffer.append(formatted_msg)
        
        # Append to main log immediately
        with open(self.log_file, "a") as f:
            f.write(formatted_msg + "\n")

    def dump(self):
        # Dump entire buffer to monitor file for easy reading
        with open(self.monitor_file, "w") as f:
            f.write("\n".join(self.buffer))


def get_curriculum_params(cycle):
    """
    Returns training parameters based on the current cycle.
    Curriculum (Phase 5.0 - Scaling):
    - Cycles 1-5: Warmup (100 sims, 200 games)
    - Cycles 6-20: Scaling (200 sims, 500 games)
    - Cycles 21+: High Quality (400 sims, 1000 games)
    """
    # Default parameters
    params = {
        'n_games': 500,
        'simulations': 200,
        'epochs': 10,
        'lr': 1e-3,
        'cpuct': 1.2,
        'temp_threshold': 15,  # Enable temperature for first 15 moves
        'noise_alpha': 0.3,    # Dirichlet noise alpha
        'noise_eps': 0.25,     # Dirichlet noise epsilon
        'opponent_type': 'self' # Default: self-play
    }
    
    if cycle <= 5:
        # Bootstrap
        params['n_games'] = 50
        params['simulations'] = 200
        params['epochs'] = 5
        params['lr'] = 1e-3
        params['cpuct'] = 1.0
        params['temp_threshold'] = 15
    elif cycle <= 10:
        # Warmup
        params['n_games'] = 100
        params['simulations'] = 200
        params['epochs'] = 10
        params['lr'] = 1e-3
        params['temp_threshold'] = 15
        params['cpuct'] = 1.0
    elif cycle <= 25:
        # Scaling
        params['n_games'] = 300
        params['simulations'] = 200
        params['epochs'] = 10
        params['lr'] = 5e-4
        params['cpuct'] = 2.0
        params['temp_threshold'] = 15
        params['noise_eps'] = 0.35
    else:
        # High Quality / Refinement
        params['n_games'] = 500
        params['simulations'] = 400
        params['epochs'] = 10
        params['lr'] = 1e-4
        params['cpuct'] = 1.5
        
    return params

def validate_cycle(current_model, previous_model_path, device, log_dir, cycle, logger=None):
    """
    Run validation matches against Heuristic and Previous Model.
    """
    msg = f"Starting validation for Cycle {cycle}..."
    if logger: logger.log(msg)
    else: print(f"\n[Validation] {msg}")
    
    # Setup players
    random_player = RandomPlayer()
    random_plus_player = RandomPlusPlayer()
    heuristic_player = HeuristicPlayer()
    min_max_mcts_player = HeuristicMinMaxMCTSPlayer()
    
    # Helper to play N games and return win rate for p1 (current_model)
    def play_validation_match(opponent_name, opponent_player, n_games=10):
        wins = 0
        draws = 0
        losses = 0
        
        msg = f"vs {opponent_name}: Playing {n_games} games..."
        if logger: logger.log(msg)
        else: print(f"[Validation] {msg}")

        for i in range(n_games):
            # Use a fresh env for validation
            env = AzulEnv()
            obs = env.reset(initial=True)
            done = False
            
            # Randomize starting player
            # Let's stick to: current_model is P0 for half, P1 for half.
            model_is_p0 = (i % 2 == 0)
            
            while not done:
                current_idx = obs['current_player']
                
                if (model_is_p0 and current_idx == 0) or (not model_is_p0 and current_idx == 1):
                    # Current Model's turn
                    # Use MCTS with low simulations for speed, or same as training?
                    # Let's use 50 sims for validation in V5 (more robust than 25)
                    mcts = MCTS(env, current_model, simulations=200, cpuct=1.0)
                    mcts.run()
                    action = mcts.select_action(temperature=0.0) # Greedy validation
                else:
                    # Opponent's turn
                    if opponent_name == "PreviousCycle":
                         # Previous model also needs MCTS
                         mcts_prev = MCTS(env, opponent_player, simulations=200, cpuct=1.0)
                         mcts_prev.run()
                         action = mcts_prev.select_action(temperature=0.0)
                    else:
                        # Heuristic/Random
                        action = opponent_player.predict(obs)
                        if not isinstance(action, tuple):
                            action = env.index_to_action(int(action))
                        
                        # Validate action to prevent crashes
                        valid_actions = env.get_valid_actions()
                        if action not in valid_actions:
                            import random
                            if valid_actions:
                                action = random.choice(valid_actions)
                            else:
                                break
                
                obs, _, done, _ = env.step(action)
            
            scores = env.get_final_scores()
            # Determine result for current_model
            p0_score, p1_score = scores
            
            if model_is_p0:
                my_score, opp_score = p0_score, p1_score
            else:
                my_score, opp_score = p1_score, p0_score
                
            if my_score > opp_score:
                wins += 1
                result_str = "WIN"
            elif my_score < opp_score:
                losses += 1
                result_str = "LOSS"
            else:
                draws += 1
                result_str = "DRAW"
            
            # Log every game result with scores
            if logger: logger.log(f"Game {i+1}/{n_games}: {result_str} ({my_score}   {opp_score})")

        
        win_rate = wins / n_games
        msg = f"vs {opponent_name}: Wins={wins}, Losses={losses}, Draws={draws} (WR: {win_rate:.2f})"
        if logger: logger.log(msg)
        else: print(f"[Validation] {msg}")
        return win_rate

    if cycle <= 10:
        wr_rival = play_validation_match("Random", random_player, n_games=10)
    elif cycle <= 25:
        wr_rival = play_validation_match("RandomPlus", random_plus_player, n_games=10)
    else:
        wr_rival = play_validation_match("Heuristic", heuristic_player, n_games=10)
    
    # 2. vs Previous Model (if exists)
    wr_previous = 0.0
    if previous_model_path and os.path.exists(previous_model_path):
        # Load previous model
        try:
            prev_checkpoint = torch.load(previous_model_path, map_location=device)
            # We need a new model instance
            # Infer shape from current model
            prev_model = copy.deepcopy(current_model) # Hack to get same architecture
            prev_model.load_state_dict(prev_checkpoint['model_state'])
            prev_model.eval()
            
            wr_previous = play_validation_match("PreviousCycle", prev_model, n_games=10)
        except Exception as e:
            msg = f"Could not load previous model: {e}"
            if logger: logger.log(msg)
            else: print(f"[Validation] {msg}")
    else:
        msg = "No previous model found. Skipping vs Previous."
        if logger: logger.log(msg)
        else: print(f"[Validation] {msg}")

    pass


def main():
    parser = argparse.ArgumentParser(description="Azul Zero Training Loop (V5 - Scaling)")
    parser.add_argument('--total_cycles', type=int, default=50, help='Number of generation-training cycles')
    parser.add_argument('--checkpoint_dir', type=str, default='data/checkpoints_v5', help='Directory to save models')
    parser.add_argument('--max_dataset_size', type=int, default=200000, help='Max examples in replay buffer')
    parser.add_argument('--resume', action='store_true', help='Resume from the latest checkpoint')
    args = parser.parse_args()

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    # Initialize Logger
    logger = TrainingLogger("logs_v5")
    
    # Setup device
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    logger.log(f"[Loop] Using device: {device}")

    # Initialize Environment to get shapes
    env = AzulEnv(num_players=2)
    obs_flat = env.encode_observation(env.reset())
    total_obs_size = obs_flat.shape[0]
    in_channels = env.num_players * 2 # 4
    spatial_size = in_channels * 5 * 5 # 100
    
    # Factories size: (N + 1) * 5
    factories_size = (env.N + 1) * 5 # 30
    
    # Global size = total - spatial - factories
    global_size = total_obs_size - spatial_size - factories_size
    action_size = env.action_size

    logger.log(f"[Loop] Shapes: Spatial={spatial_size}, Factories={factories_size}, Global={global_size}, Total={total_obs_size}")

    # Initialize Model
    model = AzulNet(
        in_channels=in_channels,
        global_size=global_size,
        action_size=action_size,
        factories_count=env.N
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    # Replay Buffer
    replay_buffer = []
    
    start_cycle = 1
    
    if args.resume:
        logger.log("[Loop] Resuming from latest checkpoint...")
        checkpoints = [f for f in os.listdir(args.checkpoint_dir) if f.startswith('model_cycle_') and f.endswith('.pt')]
        
        cycles = []
        for f in checkpoints:
            try:
                # model_cycle_X.pt
                c_str = f.replace('model_cycle_', '').replace('.pt', '')
                cycles.append(int(c_str))
            except ValueError:
                pass
        
        if cycles:
            # Sort cycles descending
            cycles.sort(reverse=True)
            
            loaded_cycle = None
            
            for c in cycles:
                ckpt_path = os.path.join(args.checkpoint_dir, f"model_cycle_{c}.pt")
                logger.log(f"[Loop] Checking checkpoint: {ckpt_path}")
                
                try:
                    checkpoint = torch.load(ckpt_path, map_location=device)
                    
                    # Temporarily load state dict to check for NaNs
                    # Ensure current model matches architecture (it should)
                    model.load_state_dict(checkpoint['model_state'])
                    
                    # Validate loaded model for NaN corruption
                    has_nan = False
                    for name, param in model.named_parameters():
                        if torch.isnan(param).any() or torch.isinf(param).any():
                            has_nan = True
                            break
                    
                    if has_nan:
                        logger.log(f"[Loop] WARNING: Checkpoint {ckpt_path} is corrupted with NaN/Inf! Trying previous...")
                        continue # Try next cycle
                    else:
                        # Found a good one
                        optimizer.load_state_dict(checkpoint['optimizer_state'])
                        start_cycle = c + 1
                        logger.log(f"[Loop] Resumed from Cycle {c}. Next cycle: {start_cycle}")
                        loaded_cycle = c
                        break
                        
                except Exception as e:
                    logger.log(f"[Loop] Failed to load checkpoint {ckpt_path}: {e}")
            
            if loaded_cycle is None:
                 logger.log("[Loop] All checkpoints corrupted or failed. Reinitializing model from scratch...")
                 model = AzulNet(
                        in_channels=in_channels,
                        global_size=global_size,
                        action_size=action_size,
                        factories_count=env.N
                    ).to(device)
                 optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
                 start_cycle = 1
                 replay_buffer = []
        else:
            logger.log("[Loop] No checkpoints found. Starting from Cycle 1.")
    else:
        logger.log("[Loop] Starting from scratch (Cycle 1).")

    for cycle in range(start_cycle, args.total_cycles + 1):
        params = get_curriculum_params(cycle)
        logger.log(f"\n=== Cycle {cycle}/{args.total_cycles} ===")
        logger.log(f"Params: {params}")
        

        
        # 1. Self-Play Generation
        logger.log(f"[Loop] Generating {params['n_games']} games (Sims: {params['simulations']})...")
        model.eval()
        new_examples, mcts_stats = generate_self_play_games(
            verbose=False,
            n_games=params['n_games'],
            env=env,
            model=model,
            simulations=params['simulations'],
            cpuct=params['cpuct'],
            temperature_threshold=params['temp_threshold'],
            noise_alpha=params['noise_alpha'],
            noise_epsilon=params['noise_eps'],
            opponent_type=params.get('opponent_type', 'self'),
            game_logger=logger
        )
        
        # Move model back to original device if needed (MPS parallel play moves to CPU)
        current_device = next(model.parameters()).device
        if current_device != device:
            logger.log(f"[Loop] Moving model back to {device} for training...")
            model = model.to(device)
        
        # Log MCTS statistics
        logger.log(f"[Loop] MCTS Stats: avg_visits={mcts_stats['avg_visits']:.1f}, "
                   f"avg_entropy={mcts_stats['avg_entropy']:.2f}, "
                   f"reuse_rate={mcts_stats['avg_reuse_rate']:.1%}, "
                   f"avg_moves={mcts_stats['avg_move_count']:.1f}")
        
        # Add to buffer
        if not new_examples:
            logger.log("[Loop] WARNING: No examples generated (all games failed?). Skipping buffer update.")
        else:
            replay_buffer.extend(new_examples)
            if len(replay_buffer) > args.max_dataset_size:
                replay_buffer = replay_buffer[-args.max_dataset_size:]
        logger.log(f"[Loop] Buffer size: {len(replay_buffer)}")
        logger.dump() # Dump after self-play
        
        # 2. Training
        logger.log(f"[Loop] Training for {params['epochs']} epochs...")
        dataset = AzulDataset(replay_buffer, augment_factories=True)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True)
        
        # Update Learning Rate
        for param_group in optimizer.param_groups:
            param_group['lr'] = params['lr']
            
        trainer = Trainer(model, optimizer, device, log_dir=f'{logger.log_dir}/cycle_{cycle}', logger=logger.log)
        history = trainer.fit(dataloader, epochs=params['epochs'])
        
        # Log training summary with detailed breakdown
        avg_train_loss = sum(history['train_loss']) / len(history['train_loss']) if history['train_loss'] else 0
        avg_policy_loss = sum(history['train_loss_policy']) / len(history['train_loss_policy']) if history['train_loss_policy'] else 0
        avg_value_loss = sum(history['train_loss_value']) / len(history['train_loss_value']) if history['train_loss_value'] else 0
        logger.log(f"[Loop] Training Finished. Avg Loss: {avg_train_loss:.4f} (Policy: {avg_policy_loss:.4f}, Value: {avg_value_loss:.4f})")
        logger.dump() # Dump after training
        
        # 3. Checkpoint
        ckpt_path = os.path.join(args.checkpoint_dir, f"model_cycle_{cycle}.pt")
        torch.save({
            'cycle': cycle,
            'model_state': model.state_dict(),
            'optimizer_state': optimizer.state_dict(),
            'params': params
        }, ckpt_path)
        logger.log(f"[Loop] Saved checkpoint to {ckpt_path}")
        
        # Save latest as 'best.pt' for easy access
        torch.save({'model_state': model.state_dict()}, os.path.join(args.checkpoint_dir, "best.pt"))

        # 4. Validation
        prev_model_path = None
        if cycle > 1:
            prev_model_path = os.path.join(args.checkpoint_dir, f"model_cycle_{cycle-1}.pt")
        
        validate_cycle(model, prev_model_path, device, f'{logger.log_dir}/cycle_{cycle}', cycle, logger=logger)
        logger.dump() # Dump after validation
if __name__ == "__main__":
    main()
