import random
import torch
import numpy as np
from .base_player import BasePlayer

class ExpertPlayer(BasePlayer):
    def __init__(self):
        super().__init__()

    def predict(self, obs):
        current_player = obs["current_player"]
        pattern_lines = np.array(obs["players"][current_player]["pattern_lines_padded"])
        wall = np.array(obs["players"][current_player]["wall"])
        valid_actions = get_valid_actions_from_obs(obs)
        
        complete_row_action = None
        one_shot_row_action = None
        some_no_foor_action = None
        best_floor_action = None
        floor_crystals = 0
        for action in valid_actions:
            flat_action = action.item()
            factory, color, row = decode_action(flat_action)
            has_color = False
            empty_slots = 0
            number_crystals_of_color = 0
            if row < 5:
                if factory < 5:
                    for i in range(len(obs["factories"][factory])):
                        if obs["factories"][factory][i] == color:
                            number_crystals_of_color += 1
                else:
                    for i in range(len(obs["center"])):
                        if obs["center"][i] == color:
                            number_crystals_of_color += 1
                for i in range(len(pattern_lines[row])):
                    if pattern_lines[row][i] == -1:
                        empty_slots += 1
                    elif pattern_lines[row][i] == color:
                        has_color = True
                if row < 5 and has_color and empty_slots == number_crystals_of_color:
                    complete_row_action = flat_action
                    break
                if row < 5 and not has_color and empty_slots == number_crystals_of_color:
                    one_shot_row_action = flat_action
                if row < 5 and empty_slots >= number_crystals_of_color:
                    some_no_foor_action = flat_action
            else:
                # Count crystals of this color on the current player's floor line
                floor_line = obs["players"][current_player]["floor_line"]
                number_crystals_of_color = 0
                for tile in floor_line:
                    if tile == color:
                        number_crystals_of_color += 1
                if best_floor_action is None or number_crystals_of_color > floor_crystals:
                    best_floor_action = flat_action
                    floor_crystals = number_crystals_of_color



        if complete_row_action is not None:
            return complete_row_action
        elif one_shot_row_action is not None:
            return one_shot_row_action
        elif some_no_foor_action is not None:
            return some_no_foor_action
        elif best_floor_action is not None:
            return best_floor_action
        else:
            print("No valid actions and center is empty.")
            print("Observation keys:", list(obs.keys()))
            raise ValueError("No fallback action possible in heuristic player")

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
