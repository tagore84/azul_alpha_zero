import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

# src/players/maximilian_times.py

import time
from copy import deepcopy

from .base_player import BasePlayer

# Minimal internal Warehouse for AI
class Warehouse:
    def __init__(self, counts):
        self.counts = counts.copy()
    @classmethod
    def from_list(cls, lst):
        return cls(list(lst))
    def get_crystals(self, color):
        return [None] * self.counts[color]
    def take_all(self, color):
        n = self.counts[color]
        self.counts[color] = 0
        return [None] * n

# Minimal internal GameState wrapper using obs
class GameState:
    """Minimal game state for MaximilianTimes AI, based on observation dict."""
    @classmethod
    def from_obs(cls, obs):
        state = cls.__new__(cls)
        state.factories = [Warehouse.from_list(f) for f in obs["factories"]]
        state.center_plate = Warehouse.from_list(obs["center"])
        state.token_warehouses = state.factories + [state.center_plate]
        state.players = obs["players"]
        state.current_player_idx = obs["current_player"]
        return state

    def get_token_warehouses(self):
        return self.token_warehouses

    def apply_move(self, src, color, row):
        wh = self.token_warehouses[src]
        crystals = wh.get_crystals(color)
        if not crystals:
            return False
        wh.take_all(color)
        return True

    def next_player(self):
        self.current_player_idx = (self.current_player_idx + 1) % len(self.players)

    def get_players(self):
        return self.players

    def get_current_player(self):
        return self.current_player_idx

C = 5  # number of colors
D = 6  # number of destinations: 5 pattern rows (0–4) + floor line (5)

def value_game_state(game_state, player, ruin_factor):
    """
    Compute a heuristic value for the given game state and player.
    """
    pts = 0.0
    clone = deepcopy(game_state)
    for p in clone.get_players():
        # Safely handle missing methods on obs-based players
        final_phase = getattr(p, 'final_phase_points', lambda: None)
        final_game = getattr(p, 'final_game_points', lambda: None)
        get_points = getattr(p, 'get_points', lambda: 0)
        is_human = getattr(p, 'is_human', lambda: False)
        final_phase()
        final_game()
        if p == player:
            pts += get_points()
        elif is_human():
            pts -= (2 * ruin_factor) * (get_points() / (len(clone.get_players()) - 1))
    return pts

class MoveTree:
    """
    Node in the move tree: stores move parameters, resulting game state,
    children, and evaluation values.
    """
    def __init__(self, src, color, row, num_crystals, state):
        self.src = src
        self.color = color
        self.row = row
        self.num_crystals = num_crystals
        self.state = state
        self.children = []
        self.parent = None
        self.value = 0.0
        self.diff = 0.0

    def add_child(self, child):
        self.children.append(child)
        child.parent = self

    def set_value(self, value):
        self.value = value

    def prune_non_maximals(self, num_to_keep):
        # Keep top `num_to_keep` children by descending value
        self.children.sort(key=lambda c: c.value, reverse=True)
        self.children = self.children[:num_to_keep]

    def get_leafs(self):
        if not self.children:
            return [self]
        leafs = []
        for c in self.children:
            leafs.extend(c.get_leafs())
        return leafs

    def calculate_diff(self, init_points, num_crystals_factor, ruin_factor, player):
        """
        Compute diff = (best leaf points - init points) + crystal bonus.
        """
        best_leaf_pts = float('-inf')
        for leaf in self.get_leafs():
            pts = value_game_state(leaf.state, player, ruin_factor)
            if pts > best_leaf_pts:
                best_leaf_pts = pts
        self.diff = (best_leaf_pts - init_points) + (self.num_crystals * num_crystals_factor)
        return self.diff

class MaximilianTimes(BasePlayer):
    """
    AI player using a limited-depth/time Minimax-like search,
    scoring moves by diff_value + crystal bonus.
    """
    def __init__(self, max_space, max_time_in_seconds, num_crystals_factor, ruin_factor):
        super().__init__()
        self.max_space = max_space
        self.max_time_in_seconds = max_time_in_seconds
        self.num_crystals_factor = num_crystals_factor
        self.ruin_factor = ruin_factor
        self.total_seconds_thinking = 0.0

    def predict(self, obs):
        original_state = GameState.from_obs(obs)
        player_idx = obs["current_player"]
        start_time = time.time()
        deadline = start_time + self.max_time_in_seconds

        # Generate root moves (no floor)
        roots = self._no_floor_moves(original_state)

        # Expand search within time limit
        for root in roots:
            self._recursive_deep_move(root, deadline, player_idx)

        # Evaluate best move by diff
        init_pts = value_game_state(deepcopy(original_state), player_idx, self.ruin_factor)
        best = None
        best_score = float('-inf')
        for root in roots:
            score = root.calculate_diff(init_pts, self.num_crystals_factor, self.ruin_factor, player_idx)
            if score > best_score:
                best_score = score
                best = root

        # If no pattern moves, choose floor move
        if best is None:
            best = self._floor_move(original_state)

        self.total_seconds_thinking += time.time() - start_time
        # Encode action index
        return best.src * (C * D) + best.color * D + best.row

    def _no_floor_moves(self, state):
        """
        Return list of MoveTree roots for all legal pattern-row moves.
        """
        moves = []
        for src, wh in enumerate(state.get_token_warehouses()):
            for color in range(C):
                crystals = wh.get_crystals(color)
                if not crystals:
                    continue
                for row in range(5):
                    clone = deepcopy(state)
                    if clone.apply_move(src, color, row):
                        mt = MoveTree(src, color, row, len(crystals), clone)
                        moves.append(mt)
        return moves

    def _recursive_deep_move(self, node, deadline, ai_idx):
        """
        Recursively expand `node` until deadline, pruning by max_space.
        """
        if time.time() >= deadline:
            return
        next_state = deepcopy(node.state)
        next_state.next_player()
        children = self._no_floor_moves(next_state)
        for child in children:
            child.set_value(value_game_state(child.state, node.state.current_player_idx, self.ruin_factor))
            node.add_child(child)
        # Keep only best children
        current_player = node.state.get_current_player()
        num_keep = 1 if current_player == ai_idx else self.max_space
        node.prune_non_maximals(num_keep)
        for child in node.children:
            self._recursive_deep_move(child, deadline, ai_idx)

    def _floor_move(self, state):
        """
        Fallback: pick the smallest available batch to floor line.
        """
        min_count = float('inf')
        choice = None
        for src, wh in enumerate(state.get_token_warehouses()):
            for color in range(C):
                cnt = len(wh.get_crystals(color))
                if 0 < cnt < min_count:
                    min_count, choice = cnt, (src, color)
        src, color = choice
        return MoveTree(src, color, 5, min_count, state)

    def get_total_seconds_thinking(self):
        return int(self.total_seconds_thinking)
