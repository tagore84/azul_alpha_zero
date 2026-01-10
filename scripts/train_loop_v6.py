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
            f.write(f"\n{'='*20}\n[{datetime.now()}] Training Session Started (V6 - Fix & Rescue)\n{'='*20}\n")

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
    Curriculum (Phase 6.0 - Fix & Rescue):
    - Cycles 1-5: Re-orientation (High Exploration, High LR)
    - Cycles 6+: Return to V5 Logic
    """
    # Default parameters (Standard V5 High Quality)
    params = {
        'n_games': 400,
        'simulations': 300,
        'epochs': 10,
        'lr': 5e-4,
        'cpuct': 1.5,
        'temp_threshold': 15, 
        'noise_alpha': 0.25,
        'noise_eps': 0.15,
        'opponent_type': 'self',
        'validation_opponent': 'MinMaxDepth1'
    }
    
    if cycle <= 5:
        # Phase 1: Re-orientation
        # High exploration and LR to unlearn "Suicide Strategy"
        # and discover positive rewards.
        params['n_games'] = 200 # Smaller batches to update network more often
        params['simulations'] = 300
        params['epochs'] = 10
        params['lr'] = 5e-4 # High LR
        params['cpuct'] = 2.0 # Very High exploration constant
        params['temp_threshold'] = 20 # Keep temperature longer
        params['noise_eps'] = 0.25 # Lots of noise
        params['noise_alpha'] = 0.3
        params['validation_opponent'] = 'RandomPlus' # Easy validation check
        
    elif cycle <= 20:
        # Phase 2: Scaling (Similar to V5 Mid-game)
        params['n_games'] = 400
        params['simulations'] = 300
        params['epochs'] = 10
        params['lr'] = 5e-4
        params['cpuct'] = 1.6
        params['temp_threshold'] = 15
        params['noise_eps'] = 0.20
        params['noise_alpha'] = 0.2
        params['validation_opponent'] = 'Heuristic'
        
    else:
        # Phase 3: Refinement (V5 Late-game)
        params['n_games'] = 500
        params['simulations'] = 400
        params['epochs'] = 10
        params['lr'] = 1e-4
        params['temp_threshold'] = 10
        params['cpuct'] = 1.2
        params['noise_eps'] = 0.10
        params['noise_alpha'] = 0.15
        params['validation_opponent'] = 'MinMaxDepth1'
        
    return params

def validate_cycle(current_model, previous_model_path, device, log_dir, cycle, params, logger=None):
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
    min_max_1 = HeuristicMinMaxMCTSPlayer(strategy='minmax', depth=1)
    min_max_2 = HeuristicMinMaxMCTSPlayer(strategy='minmax', depth=2)
    
    # Helper to play N games and return win rate for p1 (current_model)
    def play_validation_match(opponent_name, opponent_player, n_games=10):
        wins = 0
        draws = 0
        losses = 0
        
        msg = f"vs {opponent_name}: Playing {n_games} games..."
        if logger: logger.log(msg)
        else: print(f"[Validation] {msg}")

        for i in range(n_games):
            env = AzulEnv()
            obs = env.reset(initial=True)
            done = False
            
            # Randomize starting player
            model_is_p0 = (i % 2 == 0)
            
            while not done:
                current_idx = obs['current_player']
                
                if (model_is_p0 and current_idx == 0) or (not model_is_p0 and current_idx == 1):
                    # Current Model's turn
                    mcts = MCTS(env, current_model, simulations=300, cpuct=1.0, single_player_mode=True)
                    mcts.run()
                    action = mcts.select_action(temperature=0.0) # Greedy validation
                else:
                    # Opponent's turn
                    if opponent_name == "PreviousCycle":
                         mcts_prev = MCTS(env, opponent_player, simulations=300, cpuct=1.0, single_player_mode=True)
                         mcts_prev.run()
                         action = mcts_prev.select_action(temperature=0.0)
                    else:
                        action = opponent_player.predict(obs)
                        if not isinstance(action, tuple):
                            action = env.index_to_action(int(action))
                        
                        valid_actions = env.get_valid_actions()
                        if action not in valid_actions:
                            import random
                            if valid_actions:
                                action = random.choice(valid_actions)
                            else:
                                break
                
                obs, _, done, _ = env.step(action)
            
            scores = env.get_final_scores()
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
            
            if logger: logger.log(f"Game {i+1}/{n_games}: {result_str} ({my_score}   {opp_score})")
        
        win_rate = wins / n_games
        msg = f"vs {opponent_name}: Wins={wins}, Losses={losses}, Draws={draws} (WR: {win_rate:.2f})"
        if logger: logger.log(msg)
        else: print(f"[Validation] {msg}")
        return win_rate

    # Determine rival based on params
    val_opponent_name = params.get('validation_opponent', 'MinMaxDepth2')
    
    if val_opponent_name == 'Random':
        opponent_player = random_player
    elif val_opponent_name == 'RandomPlus':
        opponent_player = random_plus_player
    elif val_opponent_name == 'Heuristic':
        opponent_player = heuristic_player
    elif val_opponent_name == 'MinMaxDepth2':
        opponent_player = min_max_2
    elif val_opponent_name == 'MinMaxDepth1':
        opponent_player = min_max_1
    else:
        val_opponent_name = 'RandomPlus'
        opponent_player = random_plus_player

    wr_rival = play_validation_match(val_opponent_name, opponent_player, n_games=10)
    
    # 2. vs Previous Model (if exists)
    if previous_model_path and os.path.exists(previous_model_path):
        try:
            prev_checkpoint = torch.load(previous_model_path, map_location=device, weights_only=False)
            prev_model = copy.deepcopy(current_model)
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


def main():
    parser = argparse.ArgumentParser(description="Azul Zero Training Loop (V6 - Fix & Rescue)")
    parser.add_argument('--total_cycles', type=int, default=50, help='Number of cycles')
    parser.add_argument('--checkpoint_dir', type=str, default='data/checkpoints_v6', help='Directory to save models')
    parser.add_argument('--max_dataset_size', type=int, default=300000, help='Max replay buffer size')
    parser.add_argument('--resume', action='store_true', help='Resume V6 training if existing')
    args = parser.parse_args()

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    # Initialize Logger
    logger = TrainingLogger("logs_v6")
    
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
    in_channels = env.num_players * 2 * 5 
    spatial_size = in_channels * 5 * 5 
    factories_size = (env.N + 1) * 5
    global_size = total_obs_size - spatial_size - factories_size
    action_size = env.action_size

    logger.log(f"[Loop] Shapes: Spatial={spatial_size}, Factories={factories_size}, Global={global_size}")

    # Initialize Model
    model = AzulNet(
        in_channels=in_channels,
        global_size=global_size,
        action_size=action_size,
        factories_count=env.N
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    replay_buffer = []  # Start empty by default
    start_cycle = 1
    
    # === SCENARIO A: RESUME V6 ===
    # Check if V6 checkpoints exist
    v6_checkpoints = [f for f in os.listdir(args.checkpoint_dir) if f.startswith('model_cycle_') and f.endswith('.pt')]
    
    if args.resume and v6_checkpoints:
        logger.log("[Loop] Resume requested AND V6 checkpoints found. Resuming V6 run...")
        cycles = []
        for f in v6_checkpoints:
            try:
                c_str = f.replace('model_cycle_', '').replace('.pt', '')
                cycles.append(int(c_str))
            except ValueError: pass
            
        cycles.sort(reverse=True)
        
        for c in cycles:
            ckpt_path = os.path.join(args.checkpoint_dir, f"model_cycle_{c}.pt")
            try:
                checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
                model.load_state_dict(checkpoint['model_state'])
                
                # Load optimizer if available
                if 'optimizer_state' in checkpoint:
                     optimizer.load_state_dict(checkpoint['optimizer_state'])
                
                logger.log(f"[Loop] Loaded V6 Checkpoint Cycle {c}")
                start_cycle = c + 1
                
                # Load replay buffer
                buffer_path = os.path.join(args.checkpoint_dir, "replay_buffer.pt")
                if os.path.exists(buffer_path):
                    replay_buffer = torch.load(buffer_path, weights_only=False)
                    logger.log(f"[Loop] Loaded V6 Replay Buffer: {len(replay_buffer)} items")
                break
            except Exception as e:
                logger.log(f"[Loop] Failed to load checkpoint {ckpt_path}: {e}")
                
    else:
        # === SCENARIO B: START V6 (SOFT RESET) ===
        logger.log("[Loop] Starting V6 sequence (Soft Reset).")
        
        # Look for V5 Cycle 26 Base Model
        base_model_path = "data/checkpoints_v5/model_cycle_26.pt"
        if os.path.exists(base_model_path):
            logger.log(f"[Loop] Found V5 Base Model at {base_model_path}. Loading weights...")
            try:
                checkpoint = torch.load(base_model_path, map_location=device, weights_only=False)
                model.load_state_dict(checkpoint['model_state'])
                logger.log("[Loop] Weights loaded successfully.")
                logger.log("[Loop] WARNING: Discarding Optimizer State and Replay Buffer from V5 to force adaptation.")
                # We do NOT load replay buffer. We start empty to fill with new "Absolute Reward" examples.
            except Exception as e:
                logger.log(f"[Loop] Critical Error loading base model: {e}")
                sys.exit(1)
        else:
            logger.log(f"[Loop] WARNING: V5 Base Model NOT found at {base_model_path}.")
            logger.log("[Loop] Starting from scratch (Random weights).")
    
    # === TRAINING LOOP ===
    for cycle in range(start_cycle, args.total_cycles + 1):
        params = get_curriculum_params(cycle)
        logger.log(f"\n=== Cycle {cycle}/{args.total_cycles} ===")
        logger.log(f"Params: {params}")
        
        # 1. Self-Play Generation
        logger.log(f"[Loop] Generating {params['n_games']} games (Sims: {params['simulations']})...")
        print("." * params['n_games'])
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
        
        # Log MCTS statistics
        logger.log(f"[Loop] MCTS Stats: avg_visits={mcts_stats['avg_visits']:.1f}, "
                   f"avg_entropy={mcts_stats['avg_entropy']:.2f}, "
                   f"reuse_rate={mcts_stats.get('avg_reuse_rate', 0):.1%}, "
                   f"avg_moves={mcts_stats.get('avg_move_count', 0):.1f}")
        
        # Add to buffer
        if not new_examples:
            logger.log("[Loop] WARNING: No examples generated. Skipping buffer update.")
        else:
            replay_buffer.extend(new_examples)
            if len(replay_buffer) > args.max_dataset_size:
                replay_buffer = replay_buffer[-args.max_dataset_size:]
        logger.log(f"[Loop] Buffer size: {len(replay_buffer)}")
        logger.dump()
        
        # 2. Training
        if len(replay_buffer) < 500:
             logger.log("[Loop] Buffer too small for training (<500). Skipping training step.")
        else:
            logger.log(f"[Loop] Training for {params['epochs']} epochs...")
            dataset = AzulDataset(replay_buffer, augment_factories=True)
            dataloader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True)
            
            # Update Learning Rate
            for param_group in optimizer.param_groups:
                param_group['lr'] = params['lr']
                
            trainer = Trainer(model, optimizer, device, log_dir=f'{logger.log_dir}/cycle_{cycle}', logger=logger.log)
            history = trainer.fit(dataloader, epochs=params['epochs'])
            
            avg_train_loss = sum(history['train_loss']) / len(history['train_loss']) if history['train_loss'] else 0
            logger.log(f"[Loop] Training Finished. Avg Loss: {avg_train_loss:.4f}")
        
        logger.dump()
        
        # 3. Checkpoint
        ckpt_path = os.path.join(args.checkpoint_dir, f"model_cycle_{cycle}.pt")
        torch.save({
            'cycle': cycle,
            'model_state': model.state_dict(),
            'optimizer_state': optimizer.state_dict(),
            'params': params
        }, ckpt_path)
        logger.log(f"[Loop] Saved checkpoint to {ckpt_path}")
        
        # Save Replay Buffer
        buffer_save_path = os.path.join(args.checkpoint_dir, "replay_buffer.pt")
        try:
            torch.save(replay_buffer, buffer_save_path)
        except Exception as e:
            logger.log(f"[Loop] Failed to save replay buffer: {e}")

        # 4. Validation
        prev_model_path = None
        if cycle > 1:
            prev_model_path = os.path.join(args.checkpoint_dir, f"model_cycle_{cycle-1}.pt")
        
        validate_cycle(model, prev_model_path, device, f'{logger.log_dir}/cycle_{cycle}', cycle, params, logger=logger)
        logger.dump()

if __name__ == "__main__":
    main()
