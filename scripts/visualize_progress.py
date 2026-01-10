import re
import os
import argparse
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime

def parse_logs(log_dir):
    log_file = os.path.join(log_dir, "training_monitor.log")
    if not os.path.exists(log_file):
        print(f"Error: Log file not found at {log_file}")
        return None

    data = []
    
    current_cycle = 0
    current_cycle_stats = {
        "cycle": 0,
        "train_score_avg": None,
        "rival_win_rate": None,
        "prev_win_rate": None,
        "train_loss": None
    }
    
    # Temporary lists for current cycle
    train_scores = []
    rival_wins = 0
    rival_games = 0
    prev_wins = 0
    prev_games = 0
    
    with open(log_file, "r") as f:
        for line in f:
            # Cycle Header
            cycle_match = re.search(r"=== Cycle (\d+)/", line)
            if cycle_match:
                # Save previous cycle if valid
                if current_cycle > 0:
                    # Finalize stats for previous cycle
                    if train_scores:
                        current_cycle_stats["train_score_avg"] = sum(train_scores) / len(train_scores)
                    if rival_games > 0:
                        current_cycle_stats["rival_win_rate"] = (rival_wins / rival_games) * 100
                    if prev_games > 0:
                        current_cycle_stats["prev_win_rate"] = (prev_wins / prev_games) * 100
                    
                    data.append(current_cycle_stats.copy())
                
                # Reset for new cycle
                current_cycle = int(cycle_match.group(1))
                current_cycle_stats = {
                    "cycle": current_cycle,
                    "train_score_avg": 0,
                    "rival_win_rate": 0,
                    "prev_win_rate": 0,
                    "train_loss": 0
                }
                train_scores = []
                rival_wins = 0
                rival_games = 0
                prev_wins = 0
                prev_games = 0
                continue
                
            if current_cycle == 0:
                continue

            # Training Game Score
            train_game_match = re.search(r"\[Game \d+/\d+\] Score: (-?\d+)\s+(-?\d+),", line)
            if train_game_match:
                s0 = int(train_game_match.group(1))
                s1 = int(train_game_match.group(2))
                train_scores.extend([s0, s1]) # Self-play both are model
                continue

            # Validation Game
            val_match = re.search(r"vs (.*?): Wins=(\d+), Losses=(\d+), Draws=(\d+)", line)
            if val_match:
                opp_name = val_match.group(1)
                wins = int(val_match.group(2))
                losses = int(val_match.group(3))
                draws = int(val_match.group(4))
                total = wins + losses + draws
                
                if "Previous" in opp_name:
                    prev_wins += wins
                    prev_games += total
                else:
                    rival_wins += wins
                    rival_games += total
                continue

            # Training Loss
            loss_match = re.search(r"Training Finished\. Avg Loss: ([\d\.]+)", line)
            if loss_match:
                current_cycle_stats["train_loss"] = float(loss_match.group(1))
                continue

    # Add last cycle
    if current_cycle > 0:
        if train_scores:
            current_cycle_stats["train_score_avg"] = sum(train_scores) / len(train_scores)
        if rival_games > 0:
            current_cycle_stats["rival_win_rate"] = (rival_wins / rival_games) * 100
        if prev_games > 0:
            current_cycle_stats["prev_win_rate"] = (prev_wins / prev_games) * 100
        data.append(current_cycle_stats.copy())

    df = pd.DataFrame(data)
    return df

def plot_progress(df, output_path):
    if df is None or df.empty:
        print("No data to plot.")
        return

    plt.style.use('bmh') # Use a nice style
    fig, axes = plt.subplots(3, 1, figsize=(10, 15), sharex=True)
    
    # Plot 1: Win Rates
    cycles = df['cycle']
    if 'rival_win_rate' in df:
        axes[0].plot(cycles, df['rival_win_rate'], marker='o', label='vs Rival', color='blue')
    if 'prev_win_rate' in df:
        axes[0].plot(cycles, df['prev_win_rate'], marker='x', label='vs Prev Version', color='green', linestyle='--')
    
    axes[0].set_title('Validation Win Rates (%)')
    axes[0].set_ylabel('Win Rate %')
    axes[0].set_ylim(0, 105)
    axes[0].legend()
    axes[0].grid(True)
    
    # Plot 2: Average Training Score
    if 'train_score_avg' in df:
        axes[1].plot(cycles, df['train_score_avg'], marker='s', label='Avg Score (Self Play)', color='purple')
        axes[1].axhline(y=0, color='black', linestyle='-', alpha=0.3)
    
    axes[1].set_title('Average Self-Play Score')
    axes[1].set_ylabel('Score')
    axes[1].legend()
    axes[1].grid(True)
    
    # Plot 3: Training Loss
    if 'train_loss' in df:
        axes[2].plot(cycles, df['train_loss'], marker='d', label='Training Loss', color='red')
    
    axes[2].set_title('Training Loss')
    axes[2].set_ylabel('Loss')
    axes[2].set_xlabel('Cycle')
    axes[2].legend()
    axes[2].grid(True)
    
    plt.tight_layout()
    plt.savefig(output_path)
    print(f"Chart saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize Azul Zero Training Progress")
    parser.add_argument("--dir", type=str, default="logs_v6", help="Log directory (default: logs_v6)")
    args = parser.parse_args()
    
    df = parse_logs(args.dir)
    if df is not None:
        output_file = os.path.join(args.dir, "training_progress.png")
        plot_progress(df, output_file)
        
        # Also print latest stats
        if not df.empty:
            latest = df.iloc[-1]
            print(f"\nLatest Status (Cycle {latest['cycle']}):")
            print(f"  Win Rate vs Rival: {latest['rival_win_rate']:.1f}%")
            print(f"  Avg Score: {latest['train_score_avg']:.1f}")
