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
from players.heuristic_player import HeuristicPlayer
from players.random_player import RandomPlayer
from players.random_plus_player import RandomPlusPlayer
from mcts.mcts import MCTS
import copy

def get_curriculum_params(cycle):
    """
    Returns training parameters based on the current cycle.
    Curriculum:
    - Cycles 1-5: Fast learning (Rules & Basic Tactics)
    - Cycles 6-15: Strategic learning
    - Cycles 16+: Refinement
    """
    if cycle <= 5:
        return {
            'n_games': 50,
            'simulations': 25,
            'epochs': 5,
            'lr': 1e-3,
            'cpuct': 1.0
        }
    elif cycle <= 15:
        return {
            'n_games': 100,
            'simulations': 50,
            'epochs': 10,
            'lr': 5e-4,
            'cpuct': 1.2
        }
    else:
        return {
            'n_games': 200,
            'simulations': 100,
            'epochs': 10,
            'lr': 1e-4,
            'cpuct': 1.5
        }

def validate_cycle(current_model, previous_model_path, device, log_dir, cycle):
    """
    Run validation matches against Heuristic and Previous Model.
    """
    print(f"\n[Validation] Starting validation for Cycle {cycle}...")
    
    # Setup players
    random_player = RandomPlayer()
    random_plus_player = RandomPlusPlayer()
    heuristic_player = HeuristicPlayer()
    
    # Helper to play N games and return win rate for p1 (current_model)
    def play_validation_match(opponent_name, opponent_player, n_games=10):
        wins = 0
        draws = 0
        losses = 0
        
        # Use a fresh env for validation
        env = AzulEnv()
        
        print(f"[Validation] vs {opponent_name}: Playing {n_games} games...")
        for i in range(n_games):
            obs = env.reset(initial=True)
            done = False
            
            # Randomize starting player
            # But we want to track current_model performance.
            # Let's play 50% games as P0 and 50% as P1 if n_games is even.
            # For simplicity, let's just alternate or random.
            # Let's stick to: current_model is P0 for half, P1 for half.
            
            # Actually, let's just use the loop index. Even: Model=P0, Odd: Model=P1
            model_is_p0 = (i % 2 == 0)
            
            while not done:
                current_idx = obs['current_player']
                
                if (model_is_p0 and current_idx == 0) or (not model_is_p0 and current_idx == 1):
                    # Current Model's turn
                    # Use MCTS with low simulations for speed, or same as training?
                    # Let's use 25 sims for validation to be quick but decent.
                    mcts = MCTS(env, current_model, simulations=25, cpuct=1.0)
                    mcts.run()
                    action = mcts.select_action()
                else:
                    # Opponent's turn
                    if opponent_name == "PreviousCycle":
                         # Previous model also needs MCTS
                         mcts_prev = MCTS(env, opponent_player, simulations=25, cpuct=1.0)
                         mcts_prev.run()
                         action = mcts_prev.select_action()
                    else:
                        # Heuristic/Random
                        action = opponent_player.predict(obs)
                        if not isinstance(action, tuple):
                            action = env.index_to_action(int(action))
                
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
            
            print(f"[Validation] Game {i+1}/{n_games}: {result_str} (Model: {my_score}, Opponent: {opp_score})")
        
        win_rate = wins / n_games
        print(f"[Validation] vs {opponent_name}: Wins={wins}, Losses={losses}, Draws={draws} (WR: {win_rate:.2f})")
        return win_rate
    if cycle <= 5:
        wr_rival = play_validation_match("Random", random_player, n_games=5)
    elif cycle <= 15:
        wr_rival = play_validation_match("RandomPlus", random_plus_player, n_games=5)
    else:
        wr_rival = play_validation_match("Heuristic", heuristic_player, n_games=5)
    
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
            print(f"[Validation] Could not load previous model: {e}")
    else:
        print("[Validation] No previous model found. Skipping vs Previous.")

    # Log to file/tensorboard if needed (trainer writer is not passed here easily, but we can print)
    with open(os.path.join(log_dir, "validation_results.txt"), "a") as f:
        f.write(f"Cycle {cycle}: vs Rival WR={wr_rival:.2f}, vs Previous WR={wr_previous:.2f}\n")


def main():
    parser = argparse.ArgumentParser(description="Azul Zero Training Loop")
    parser.add_argument('--total_cycles', type=int, default=20, help='Number of generation-training cycles')
    parser.add_argument('--checkpoint_dir', type=str, default='data/checkpoints', help='Directory to save models')
    parser.add_argument('--max_dataset_size', type=int, default=10000, help='Max examples in replay buffer')
    parser.add_argument('--resume', action='store_true', help='Resume from the latest checkpoint')
    args = parser.parse_args()

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    # Setup device
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    print(f"[Loop] Using device: {device}")

    # Initialize Environment to get shapes
    env = AzulEnv(num_players=2)
    obs_flat = env.encode_observation(env.reset())
    total_obs_size = obs_flat.shape[0]
    in_channels = env.num_players * 2
    spatial_size = in_channels * 5 * 5
    global_size = total_obs_size - spatial_size
    action_size = env.action_size

    # Initialize Model
    model = AzulNet(in_channels, global_size, action_size).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    # Replay Buffer
    replay_buffer = []
    
    start_cycle = 1
    
    if args.resume:
        # Find latest checkpoint
        checkpoints = [f for f in os.listdir(args.checkpoint_dir) if f.startswith('model_cycle_') and f.endswith('.pt')]
        if checkpoints:
            # Extract cycle numbers
            cycles = []
            for cp in checkpoints:
                try:
                    c = int(cp.split('_')[2].split('.')[0])
                    cycles.append(c)
                except (IndexError, ValueError):
                    pass
            
            if cycles:
                latest_cycle = max(cycles)
                ckpt_path = os.path.join(args.checkpoint_dir, f"model_cycle_{latest_cycle}.pt")
                print(f"[Loop] Resuming from checkpoint: {ckpt_path}")
                
                checkpoint = torch.load(ckpt_path, map_location=device)
                model.load_state_dict(checkpoint['model_state'])
                optimizer.load_state_dict(checkpoint['optimizer_state'])
                start_cycle = checkpoint['cycle'] + 1
                print(f"[Loop] Resumed. Next cycle: {start_cycle}")
            else:
                print("[Loop] No valid checkpoints found to resume from. Starting from scratch.")
        else:
            print("[Loop] No checkpoints found to resume from. Starting from scratch.")

    for cycle in range(start_cycle, args.total_cycles + 1):
        params = get_curriculum_params(cycle)
        print(f"\n=== Cycle {cycle}/{args.total_cycles} ===")
        print(f"Params: {params}")
        
        # 1. Self-Play Generation
        print(f"[Loop] Generating {params['n_games']} games (Sims: {params['simulations']})...")
        model.eval()
        new_examples = generate_self_play_games(
            verbose=False,
            n_games=params['n_games'],
            env=env,
            model=model,
            simulations=params['simulations'],
            cpuct=params['cpuct']
        )
        
        # Add to buffer
        replay_buffer.extend(new_examples)
        if len(replay_buffer) > args.max_dataset_size:
            replay_buffer = replay_buffer[-args.max_dataset_size:]
        print(f"[Loop] Buffer size: {len(replay_buffer)}")
        
        # 2. Training
        print(f"[Loop] Training for {params['epochs']} epochs...")
        dataset = AzulDataset(replay_buffer, augment_factories=True)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True)
        
        # Update Learning Rate
        for param_group in optimizer.param_groups:
            param_group['lr'] = params['lr']
            
        trainer = Trainer(model, optimizer, device, log_dir=f'logs/cycle_{cycle}')
        trainer.fit(dataloader, epochs=params['epochs'])
        
        # 3. Checkpoint
        ckpt_path = os.path.join(args.checkpoint_dir, f"model_cycle_{cycle}.pt")
        torch.save({
            'cycle': cycle,
            'model_state': model.state_dict(),
            'optimizer_state': optimizer.state_dict(),
            'params': params
        }, ckpt_path)
        print(f"[Loop] Saved checkpoint to {ckpt_path}")
        
        # Save latest as 'best.pt' for easy access
        torch.save({'model_state': model.state_dict()}, os.path.join(args.checkpoint_dir, "best.pt"))

        # 4. Validation
        prev_model_path = None
        if cycle > 1:
            prev_model_path = os.path.join(args.checkpoint_dir, f"model_cycle_{cycle-1}.pt")
        
        validate_cycle(model, prev_model_path, device, f'logs/cycle_{cycle}', cycle)

if __name__ == "__main__":
    main()
