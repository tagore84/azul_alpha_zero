
import re
import numpy as np

log_file = "logs_v5/training_monitor.log"

def parse_logs():
    train_scores = []
    val_scores = []
    
    with open(log_file, 'r') as f:
        lines = f.readlines()
        
    in_training = False
    in_validation = False
    
    for line in lines:
        if "Generating 100 games" in line:
            in_training = True
            in_validation = False
            continue
            
        if "Starting validation" in line:
            in_training = False
            in_validation = True
            continue
            
        if in_training and "[Game" in line:
            # [Game 1/100] Score: -112--44, ... Winner: P1 (Agent), ...
            match = re.search(r"Score: (-?\d+)--(-?\d+),.*Winner: (P\d) \(Agent\)", line)
            if match:
                s0 = int(match.group(1))
                s1 = int(match.group(2))
                winner_p = match.group(3) # P0 or P1
                
                if winner_p == "P0":
                    agent_score = s0
                else:
                    agent_score = s1
                train_scores.append(agent_score)
                
        if in_validation and "Game " in line and "WIN" in line or "LOSS" in line or "DRAW" in line:
            # Game 1/10: WIN (-204--207)
            # Validation usually P0 is the "First Player" aka the Agent being evaluated?
            # Need to confirm if validation randomizes sides.
            # Assuming standard Arena: Agent vs Random. 
            # If win rate is reported as Agent wins, then we need to know which score is Agent.
            # Usually Arena logs Score: Agent - Opponent. 
            # Logs say: "WIN (-204--207)".
            # If Agent won, and scores are -204 and -207. 
            # -204 > -207. So -204 is the winner (Agent).
            
            match = re.search(r"\((-?\d+)--(-?\d+)\)", line)
            if match:
                s0 = int(match.group(1))
                s1 = int(match.group(2))
                # Assuming Arena logs [Model] vs [Opponent].
                # So P0 is Agent.
                val_scores.append(s0)

    print(f"Training Games (MCTS): {len(train_scores)}")
    print(f"Avg Training Score: {np.mean(train_scores):.2f}")
    print(f"Min/Max Training: {np.min(train_scores)} / {np.max(train_scores)}")
    
    print(f"\nValidation Games (Network): {len(val_scores)}")
    print(f"Avg Validation Score: {np.mean(val_scores):.2f}")
    if len(val_scores) > 0:
        print(f"Min/Max Validation: {np.min(val_scores)} / {np.max(val_scores)}")

if __name__ == "__main__":
    parse_logs()
