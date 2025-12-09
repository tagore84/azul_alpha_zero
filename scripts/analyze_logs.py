import re
import statistics
from collections import defaultdict

LOG_FILE = "logs_v5/training_monitor.log"

def parse_logs():
    stats = defaultdict(lambda: {
        "train_games": 0,
        "train_max_rounds": 0,
        "train_scores_p0": [],
        "train_scores_p1": [],
        
        "val_rival_games": 0,
        "val_rival_scores_p0": [],
        "val_rival_scores_p1": [],
        "val_rival_wins": 0,
        "val_rival_losses": 0,

        "val_prev_games": 0,
        "val_prev_scores_p0": [],
        "val_prev_scores_p1": [],
        "val_prev_wins": 0,
        "val_prev_losses": 0,

        "train_loss_policy": None,
        "train_loss_value": None
    })
    
    current_cycle = 0
    current_opponent = None  # "Rival" or "Prev"

    with open(LOG_FILE, "r") as f:
        for line in f:
            # Detect Cycle Header
            cycle_match = re.search(r"=== Cycle (\d+)/", line)
            if cycle_match:
                current_cycle = int(cycle_match.group(1))
                current_opponent = None
                continue
            
            if current_cycle == 0:
                continue

            # Parse Training Game
            # [2025-12-08 16:36:35] [Game 1/50] Score: -11--64, Rounds: 7, Moves: 76, Winner: P0, End: normal_end
            train_game_match = re.search(r"\[Game \d+/\d+\] Score: (-?\d+)-(-?\d+),.*End: (\w+)", line)
            if train_game_match:
                s0 = int(train_game_match.group(1))
                s1 = int(train_game_match.group(2))
                end_reason = train_game_match.group(3)
                
                stats[current_cycle]["train_games"] += 1
                stats[current_cycle]["train_scores_p0"].append(s0)
                stats[current_cycle]["train_scores_p1"].append(s1)
                
                if end_reason == "max_rounds":
                    stats[current_cycle]["train_max_rounds"] += 1
                continue

            # Parse Training Loss
            # [Loop] Training Finished. Avg Loss: 3.8096 (Policy: 3.6685, Value: 0.1411)
            loss_match = re.search(r"Training Finished.*Policy: ([\d\.]+), Value: ([\d\.]+)", line)
            if loss_match:
                stats[current_cycle]["train_loss_policy"] = float(loss_match.group(1))
                stats[current_cycle]["train_loss_value"] = float(loss_match.group(2))
                continue

            # Detect Validation Opponent
            # [2025-12-08 17:49:15] vs Random: Playing 10 games...
            # [2025-12-08 23:23:20] vs Random: Wins=10, Losses=0, Draws=0 (WR: 1.00)
            # [2025-12-09 00:00:56] vs PreviousCycle: Playing 10 games...
            
            # We look for "vs X: Playing" to switch context
            opponent_match = re.search(r"vs (.*?): Playing", line)
            if opponent_match:
                opp_name = opponent_match.group(1)
                if "Previous" in opp_name:
                    current_opponent = "Prev"
                else:
                    # Random, RandomPlus, Heuristic -> Rival
                    current_opponent = "Rival"
                continue

            # Parse Validation Game
            # [2025-12-08 17:49:15] Game 1/10: LOSS (-96-91)
            val_game_match = re.search(r"Game \d+/\d+: (WIN|LOSS|DRAW) \((-?\d+)-(-?\d+)\)", line)
            if val_game_match:
                if current_opponent is None:
                    continue  # Should not happen if log is consistent

                outcome = val_game_match.group(1)
                s0 = int(val_game_match.group(2))
                s1 = int(val_game_match.group(3))
                
                prefix = f"val_{current_opponent.lower()}"
                
                stats[current_cycle][f"{prefix}_games"] += 1
                stats[current_cycle][f"{prefix}_scores_p0"].append(s0)
                stats[current_cycle][f"{prefix}_scores_p1"].append(s1)
                
                if outcome == "WIN":
                    stats[current_cycle][f"{prefix}_wins"] += 1
                elif outcome == "LOSS":
                    stats[current_cycle][f"{prefix}_losses"] += 1
                continue

    output_lines = []
    # Header
    # Cycle | MaxRounds | AvgSc(Tr) | AvgSc(Riv) | WR(Riv) | WR(Prev) | Val Loss
    header = f"{'Cycle':<6} | {'MaxRounds':<10} | {'Avg Sc (Tr)':<12} | {'Avg Sc (Riv)':<12} | {'WR (Riv)':<8} | {'WR (Prev)':<8} | {'Val Loss':<8}"
    output_lines.append(header)
    output_lines.append("-" * 90)
    
    for cycle in sorted(stats.keys()):
        d = stats[cycle]
        
        # Training Stats
        n_train = d["train_games"]
        mr_count = d["train_max_rounds"]
        mr_pct = (mr_count / n_train * 100) if n_train > 0 else 0
        
        all_train_scores = d["train_scores_p0"] + d["train_scores_p1"]
        avg_train_score = statistics.mean(all_train_scores) if all_train_scores else 0
        
        # Validation Stats: Rival
        all_riv_scores = d["val_rival_scores_p0"] + d["val_rival_scores_p1"]
        avg_riv_score = statistics.mean(all_riv_scores) if all_riv_scores else 0
        n_riv = d["val_rival_games"]
        riv_wr = (d["val_rival_wins"] / n_riv * 100) if n_riv > 0 else 0
        
        # Validation Stats: Previous
        n_prev = d["val_prev_games"]
        prev_wr = (d["val_prev_wins"] / n_prev * 100) if n_prev > 0 else 0
        
        val_loss_str = f"{d['train_loss_value']:.4f}" if d['train_loss_value'] is not None else "N/A"
        
        line_str = f"{cycle:<6} | {mr_count}/{n_train} ({mr_pct:.0f}%) | {avg_train_score:<12.2f} | {avg_riv_score:<12.2f} | {riv_wr:<6.1f}%  | {prev_wr:<6.1f}%  | {val_loss_str:<8}"
        output_lines.append(line_str)

    # Print to console
    print("\n".join(output_lines))

    # Write to file
    output_file = "logs_v5/training_analyzed.log"
    with open(output_file, "w") as f:
        f.write("\n".join(output_lines) + "\n")
    print(f"\nAnalysis saved to {output_file}")

if __name__ == "__main__":
    parse_logs()
