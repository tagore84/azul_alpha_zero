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
    # Cycle | MaxRounds | Diff (Tr)(Std) | Diff (Riv)(Std) | Diff (Prev)(Std) | WR (Riv) | WR (Prev)
    header = f"{'Cycle':<6} | {'MaxRounds':<10} | {'Diff (Tr) (Std)':<18} | {'Diff (Riv) (Std)':<18} | {'Diff (Prev) (Std)':<18} | {'WR (Riv)':<8} | {'WR (Prev)':<8}"
    output_lines.append(header)
    output_lines.append("-" * 130)
    
    for cycle in sorted(stats.keys()):
        d = stats[cycle]
        
        # Training Stats (Self-Play)
        n_train = d["train_games"]
        mr_count = d["train_max_rounds"]
        mr_pct = (mr_count / n_train * 100) if n_train > 0 else 0
        
        # Calculate Diff P0 - P1 (First Player Advantage?)
        # Since it's self play, we should probably look at abs diff (margin) or just diff.
        # User asked for "maximizing point difference".
        # In self-play this is ambiguous. I will show P0-P1 to see balance.
        # Or I will show Abs(P0-P1) as "Intensity"? 
        # Let's show P0-P1.
        train_diffs = [p0 - p1 for p0, p1 in zip(d["train_scores_p0"], d["train_scores_p1"])]
        avg_tr, std_tr = calc_stats(train_diffs)
        tr_str = f"{avg_tr:.1f} ({std_tr:.1f})"
        
        # Validation Stats: Rival (Model - Rival)
        # Note: In validation logic, 's0' is always Model? No, we need to check log parsing.
        # In validate_cycle: "Current Model's turn... if (model_is_p0 and current_idx == 0)..."
        # The log says: "WIN (My-Opp)" or "LOSS (My-Opp)".
        # The regex captured (s0)-(s1) from "WIN (96-91)".
        # The logging line in validation is: logger.log(f"Game ... {result_str} ({my_score}-{opp_score})")
        # So group(2) is ALWAYS MyScore, group(3) is ALWAYS OppScore.
        # Verified in train_loop_v5.py:
        # if my_score > opp_score ... logger.log(f"... ({my_score}-{opp_score})")
        
        # Riv Diffs
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
        
        # line_str = f"{cycle:<6} | {mr_count}/{n_train} ({mr_pct:.0f}%) | {train_str:<18} | {riv_str:<18} | {riv_wr:<6.1f}%  | {prev_wr:<6.1f}%  | {val_loss_str:<8}"
        line_str = f"{cycle:<6} | {mr_count}/{n_train} ({mr_pct:.0f}%) | {tr_str:<18} | {riv_str:<18} | {prev_str:<18} | {riv_wr:<6.1f}%  | {prev_wr:<6.1f}%"
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
