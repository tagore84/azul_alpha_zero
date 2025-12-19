# src/players/heuristic_player.py
import random
import torch
import numpy as np
import copy
from azul.rules import (
    validate_origin,
    place_on_pattern_line,
    transfer_to_wall,
    calculate_floor_penalization,
    calculate_final_bonus,
    Color
)

class HeuristicPlayer:
    def __init__(self):
        self.device = torch.device("cpu")

    def predict(self, obs):
        current_player = obs["current_player"]
        valid_actions = get_valid_actions_from_obs(obs)

        best_score = float("-inf")
        best_action = None
        
        # Pre-calculate current bonuses to know delta
        # But for efficiency, we can just check what the state WOULD be
        
        indices = valid_actions.tolist()
        # Shuffle indices to break ties randomly
        random.shuffle(indices)

        for flat_action in indices:
            # Simulate the action to get accurate score delta
            immediate_score, next_wall, next_floor = self._simulate_score(obs, flat_action)
            
            # Calculate strategic bonus based on the resulting state
            strategic_bonus = self._get_strategic_bonus(obs, next_wall, next_floor, flat_action)
            
            total_score = immediate_score + strategic_bonus
            
            # print(f"Action {decode_action(flat_action)} -> Score: {immediate_score}, Strat: {strategic_bonus}, Total: {total_score}")

            if total_score > best_score:
                best_score = total_score
                best_action = flat_action

        if best_action is not None:
             # Ensure we return a single integer or tuple as expected? 
             # The previous code returned `action.item()` which is an int if tensor, or just int.
             # Env.step expects tuple if we convert it, but debug_game handles int too.
             # Let's return the flat action integer.
            return best_action
        else:
             # Fallback (should not happen if valid_actions is correct)
             # Reuse fallback logic but cleaner
             return self._fallback_action(obs)

    def _simulate_score(self, obs, flat_action):
        """
        Simulates the action and returns:
        - immediate_score_delta: Points gained/lost this turn (including penalties)
        - next_wall: The state of the wall after the move (for strategic checks)
        - next_floor: The state of the floor after the move
        """
        factory, color, row = decode_action(flat_action)
        current_player = obs["current_player"]
        p_obs = obs["players"][current_player]
        
        # Clone state to simulate
        pattern_lines = [np.array(pl) for pl in p_obs["pattern_lines"]]
        wall = np.array(p_obs["wall"])
        floor_line = np.array(p_obs["floor_line"])
        
        # 1. Determine tiles taken
        if factory < len(obs["factories"]):
            count = obs["factories"][factory][color]
            # Taking from factory: no floor penalty from center token
        else:
            count = obs["center"][color]
            # Taking from center: if first token is there, take it
            if obs["first_player_token"]:
                # Logic: Add -1 penalty token to floor
                # Find first empty slot
                idxs = np.where(floor_line == -1)[0]
                if idxs.size > 0:
                    floor_line[idxs[0]] = 5 # arbitrary marker for penalty token
        
        current_penalty = calculate_floor_penalization(floor_line)
        
        # 2. Place tiles
        speculative_points = 0
        
        if row < 5:
            # Pattern line placement
            line = pattern_lines[row]
            # Check if line has different color (should be handled by valid_actions, but good to be safe)
            # We assume valid_actions is correct.
            
            new_line, overflow = place_on_pattern_line(line, color, count)
            pattern_lines[row] = new_line
            
            # Check if line is full
            if -1 not in new_line:
                # Simulate moving to wall
                # Note: rules.transfer_to_wall modifies wall in place and returns points
                points = transfer_to_wall(wall.tolist(), new_line.tolist(), row)
                 # Update numpy wall from list
                # Actually transfer_to_wall modifies the list passed to it. 
                # Our `wall` is numpy, so we need to be careful.
                # Let's use a temp list wall.
                temp_wall_list = wall.tolist()
                points = transfer_to_wall(temp_wall_list, new_line.tolist(), row)
                wall = np.array(temp_wall_list)
                
                speculative_points += points
                pattern_lines[row][:] = -1 # Clear line (simplification for next state view)

            # Handle overflow
            for _ in range(overflow):
                idxs = np.where(floor_line == -1)[0]
                if idxs.size > 0:
                     floor_line[idxs[0]] = color
        else:
            # Direct to floor
            for _ in range(count):
                idxs = np.where(floor_line == -1)[0]
                if idxs.size > 0:
                     floor_line[idxs[0]] = color
        
        # 3. Calculate new penalty
        new_penalty = calculate_floor_penalization(floor_line)
        penalty_delta = new_penalty - current_penalty
        
        total_delta = speculative_points + penalty_delta
        
        return total_delta, wall, floor_line


    def _get_strategic_bonus(self, obs, wall, floor_line, flat_action):
        """
        Calculates a bonus score for strategic value beyond immediate points.
        """
        bonus = 0
        factory, color, row = decode_action(flat_action)
        
        # 1. Avoid taking first player token if penalty is too high?
        # Already accounted for in immediate score (penalty delta).
        # But we might want to encourage taking it if it's "cheap" to go first next turn.
        if factory == len(obs["factories"]) and obs["first_player_token"]:
             # If penalty is small (-1 or -2), it might be worth it.
             # Let's say +1.5 strategic value for controlling next turn
             bonus += 1.5
        
        if row < 5:
             # 2. Encourage filling columns (7 points end game)
             # Check if this placement helps fill a column
             # We need to know which column this color corresponds to in this row
             # Standard Azul board: diagonal shift.
             col = (color + row) % 5
             # Count tiles in this column
             col_tiles = (wall[:, col] != -1).sum()
             if col_tiles == 4:
                 bonus += 5 # Massive bonus for completing a column (or being close)
             elif col_tiles == 3:
                 bonus += 2
             
             # 3. Encourage filling colors (10 points end game)
             color_tiles = (wall == color).sum()
             if color_tiles == 4:
                 bonus += 6
             elif color_tiles == 3:
                 bonus += 2

             # 4. Encourage filling rows (2 points end game) - usually immediate points handle this, 
             # but completing a row also ends the game, which might be good or bad.
             # If we are winning, valid.
        
        # 5. Penalize filling floor line too much (risk of disaster)
        # Immediate penalty handles the score, but we want to avoid being *close* to disaster?
        # Maybe redundant.
        
        return bonus

    def _fallback_action(self, obs):
        N = len(obs["factories"])
        C = 5
        for source_idx in range(N + 1):
            source = obs["factories"][source_idx] if source_idx < N else obs["center"]
            for color in range(C):
                if source[color] > 0:
                    return source_idx * (C * 6) + color * 6 + 5
        raise ValueError("No fallback action possible")

def decode_action(index):
    """
    Decode an action index into (factory/source index, color, destination row).
    Actions are encoded as: index = source_idx * (C * 6) + color * 6 + dest.
    C is number of colors (5), dest count is 6 (pattern lines 0–4 and floor as 5).
    """
    C = 5  # number of colors
    D = 6  # number of possible destinations (5 pattern rows + floor)
    factory = index // (C * D)
    remainder = index % (C * D)
    color = remainder // D
    row = remainder % D
    return factory, color, row

def is_row_closable(row):
    return (row != -1).sum().item() == len(row) - 1

def is_row_fillable(row):
    return (row != -1).any().item()

def get_valid_actions_from_obs(obs):
    C = 5  # number of colors
    N = len(obs["factories"])  # number of factories
    valid_actions = []
    current_player = obs["current_player"]
    wall = obs["players"][current_player]["wall"]

    for source_idx in range(N + 1):
        source = obs["factories"][source_idx] if source_idx < N else obs["center"]
        for color in range(C):
            if source[color] == 0:
                continue
            for dest in range(6):
                if dest < 5:
                    wall_row = wall[dest]
                    if color in wall_row:
                        continue
                index = source_idx * (C * 6) + color * 6 + dest
                valid_actions.append(index)
    return torch.tensor(valid_actions)
