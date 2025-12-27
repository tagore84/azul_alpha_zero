import re
import statistics
from collections import defaultdict

LOG_FILE = "logs_v5/training_monitor.log"

def calc_stats(scores):
    if not scores:
        return 0, 0
    mean_val = statistics.mean(scores)
    stdev_val = statistics.stdev(scores) if len(scores) > 1 else 0
    return mean_val, stdev_val

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
        "train_loss_value": None,
        
        "train_rounds": []
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
            # Supports "Score: 10  20" (current)
            # Fixed regex to not eat negative sign of second number
            # Also captures Rounds: 5
            train_game_match = re.search(r"\[Game \d+/\d+\] Score: (-?\d+)\s+(-?\d+),.*Rounds: (\d+),.*End: (\w+)", line)
            if train_game_match:
                s0 = int(train_game_match.group(1))
                s1 = int(train_game_match.group(2))
                rounds = int(train_game_match.group(3))
                end_reason = train_game_match.group(4)
                
                stats[current_cycle]["train_games"] += 1
                stats[current_cycle]["train_scores_p0"].append(s0)
                stats[current_cycle]["train_scores_p1"].append(s1)
                stats[current_cycle]["train_rounds"].append(rounds)
                
                if "max_rounds" in end_reason:
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
            # Supports "LOSS (-96-91)" and "LOSS (-96   91)"
            val_game_match = re.search(r"Game \d+/\d+: (WIN|LOSS|DRAW) \((-?\d+)\s+(-?\d+)\)", line)
            if val_game_match:
                if current_opponent is None:
                    continue  

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
    # Cycle | MaxRounds | AvgRounds (Tr) | AvgScore (Tr) | AvgScore (Riv) | Diff (Tr)(Std) | Diff (Riv)(Std) | Diff (Prev)(Std) | WR (Riv) | WR (Prev)
    header = f"{'Cycle':<6} | {'MaxRounds':<10} | {'AvgRounds (Tr)':<15} | {'AvgScore (Tr)':<15} | {'AvgScore (Riv)':<15} | {'Diff (Tr) (Std)':<18} | {'Diff (Riv) (Std)':<18} | {'Diff (Prev) (Std)':<18} | {'WR (Riv)':<8} | {'WR (Prev)':<8}"
    output_lines.append(header)
    output_lines.append("-" * 166)
    
    for cycle in sorted(stats.keys()):
        d = stats[cycle]
        
        # Training Stats (Self-Play)
        n_train = d["train_games"]
        mr_count = d["train_max_rounds"]
        mr_pct = (mr_count / n_train * 100) if n_train > 0 else 0
        
        # Avg Rounds
        avg_rounds = statistics.mean(d["train_rounds"]) if d["train_rounds"] else 0
        rounds_str = f"{avg_rounds:.1f}"
        
        # Avg Score (Combined P0 + P1 since it's self play)
        all_train_scores = d["train_scores_p0"] + d["train_scores_p1"]
        avg_sc, std_sc = calc_stats(all_train_scores)
        sc_str = f"{avg_sc:.1f} ({std_sc:.1f})"

        # Calculate Diff P0 - P1 
        train_diffs = [p0 - p1 for p0, p1 in zip(d["train_scores_p0"], d["train_scores_p1"])]
        avg_tr, std_tr = calc_stats(train_diffs)
        tr_str = f"{avg_tr:.1f} ({std_tr:.1f})"
        
        # Validation Stats: Rival (Model - Rival)
        # s0 is Model, s1 is Rival
        riv_scores_model = d["val_rival_scores_p0"]
        avg_riv_sc, std_riv_sc = calc_stats(riv_scores_model)
        riv_sc_str = f"{avg_riv_sc:.1f} ({std_riv_sc:.1f})"
        
        riv_diffs = [p0 - p1 for p0, p1 in zip(d["val_rival_scores_p0"], d["val_rival_scores_p1"])]
        avg_riv, std_riv = calc_stats(riv_diffs)
        riv_str = f"{avg_riv:.1f} ({std_riv:.1f})"

        n_riv = d["val_rival_games"]
        riv_wr = (d["val_rival_wins"] / n_riv * 100) if n_riv > 0 else 0
        
        # Prev Diffs
        prev_diffs = [p0 - p1 for p0, p1 in zip(d["val_prev_scores_p0"], d["val_prev_scores_p1"])]
        avg_prev, std_prev = calc_stats(prev_diffs)
        prev_str = f"{avg_prev:.1f} ({std_prev:.1f})"
        
        n_prev = d["val_prev_games"]
        prev_wr = (d["val_prev_wins"] / n_prev * 100) if n_prev > 0 else 0
        
        line_str = f"{cycle:<6} | {mr_count}/{n_train} ({mr_pct:.0f}%) | {rounds_str:<15} | {sc_str:<15} | {riv_sc_str:<15} | {tr_str:<18} | {riv_str:<18} | {prev_str:<18} | {riv_wr:<6.1f}%  | {prev_wr:<6.1f}%"
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
