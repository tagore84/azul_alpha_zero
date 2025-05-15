import random
import torch
import numpy as np

class HeuristicPlayer:
    def __init__(self):
        self.device = torch.device("cpu")

    def predict(self, obs):
        current_player = obs["current_player"]
        pattern_lines = np.array(obs["players"][current_player]["pattern_lines_padded"])
        wall = np.array(obs["players"][current_player]["wall"])
        valid_actions = get_valid_actions_from_obs(obs)

        best_score = float("-inf")
        best_action = None

        for action in valid_actions:
            flat_action = action.item()
            factory, color, row = decode_action(flat_action)

            score = 0

            if row < 5:
                pline = pattern_lines[row]
                if color in wall[row]:
                    continue  # ilegal, pero por si acaso
                filled = (pline != -1).sum()
                total = len(pline)
                if color in pline or np.all(pline == -1):
                    score += 10  # colocar color compatible o en vacío
                    score += 5 * filled  # más llena la línea = mejor
                    if filled == total - 1:
                        score += 20  # ¡completa!
            else:
                score -= 15  # penaliza ir al suelo

            # pequeñas bonificaciones por usar fábricas y no centro
            if factory < 5:
                score += 1

            if score > best_score:
                best_score = score
                best_action = flat_action

        if best_action is not None:
            return best_action
        else:
            N = len(obs["factories"])
            C = 5  # number of colors
            for source_idx in range(N + 1):
                source = obs["factories"][source_idx] if source_idx < N else obs["center"]
                for color in range(C):
                    if source[color] > 0:
                        return source_idx * (C * 6) + color * 6 + 5
            # Si incluso eso falla, lanzamos excepción
            print("No valid actions and center is empty.")
            print("Observation keys:", list(obs.keys()))
            raise ValueError("No fallback action possible in heuristic player")

def decode_action(index):
    # Sustituye esta función por la real usada en tu entorno
    factory = index // 25
    color = (index % 25) // 5
    row = index % 5
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
