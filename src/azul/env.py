# src/azul/env.py

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Tuple

from azul.utils import print_floor, print_wall
from .rules import validate_origin, place_on_pattern_line, transfer_to_wall, calculate_floor_penalization, calculate_final_bonus, Color
import random  # Añade esto al principio del archivo
import copy  # Add this import at the top of the file if not present

class AzulEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, num_players: int = 2, factories_count: int = 5, seed: int = None, max_rounds: int = 1000):
        super().__init__()
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
        self.num_players = num_players
        self.C: int = len(Color)
        self.N: int = factories_count
        self.max_rounds: int = max_rounds
        self.L_floor: int = 7
        self.L_floor: int = 7

        # Game state
        self.bag: np.ndarray = np.full(self.C, 20, dtype=int)
        self.discard: np.ndarray = np.zeros(self.C, dtype=int)
        self.factories: np.ndarray = np.zeros((self.N, self.C), dtype=int)
        self.center: np.ndarray = np.zeros(self.C, dtype=int)
        self.first_player_token: bool = False
        self.first_player_next_round: int = -1 # -1 means not yet taken

        # Players state
        self.players = [
            {
                'pattern_lines': [np.full(i+1, -1, dtype=int) for i in range(5)],
                'wall': np.full((5, 5), -1, dtype=int),
                'floor_line': np.full(self.L_floor, -1, dtype=int),
                'score': 0
            }
            for _ in range(self.num_players)
        ]
        self.current_player: int = 0
        self.round_count: int = 1

        # Action: (source_idx 0..N-1 factories, N=center), color 0..C-1, dest 0..5 (pattern lines 0-4, 5=floor)
        self.action_space = spaces.Tuple((
            spaces.Discrete(self.N + 1),
            spaces.Discrete(self.C),
            spaces.Discrete(6)
        ))
        # Flattened action representation size
        self.action_size = (self.N + 1) * self.C * 6

        # Define observation space
        self.observation_space = spaces.Dict({
            'bag': spaces.Box(low=0, high=20, shape=(self.C,), dtype=int),
            'discard': spaces.Box(low=0, high=100, shape=(self.C,), dtype=int),
            'factories': spaces.Box(low=0, high=4, shape=(self.N, self.C), dtype=int),
            'center': spaces.Box(low=0, high=4, shape=(self.C,), dtype=int),
            'first_player_token': spaces.Discrete(2),
            'players_pattern_lines': spaces.Box(low=-1, high=self.C-1, shape=(self.num_players, 5, 5), dtype=int),
            'players_wall': spaces.Box(low=-1, high=self.C-1, shape=(self.num_players, 5, 5), dtype=int),
            'players_floor_line': spaces.Box(low=-1, high=self.C, shape=(self.num_players, self.L_floor), dtype=int),
            'players_score': spaces.Box(low=-1000, high=1000, shape=(self.num_players,), dtype=int),
            'current_player': spaces.Discrete(self.num_players)
        })

        # Initialize game
        self.round_accumulated_score = [0] * self.num_players
        self.reset(initial=True) # Ensure initial reset sets everything
        self.done = False

    def get_winner(self):
        """
        Returns the index of the player with the highest score in array.
        If there is a tie, returns the index of all winners.
        """
        winners = []
        if self.done:
            scores = [p['score'] for p in self.players]
            max_score = max(scores)
            winners = [i for i, score in enumerate(scores) if score == max_score]
        return winners

    def reset(self, initial: bool = False):
        # Reset bag and discard
        self.bag[:] = 20
        self.discard[:] = 0
        
        # Reset accumulated score logic
        self.round_accumulated_score = [0] * self.num_players

        # Reset player states
        for p in self.players:
            p['pattern_lines'] = [np.full(i+1, -1, dtype=int) for i in range(5)]
            if initial:  # ✅ solo al principio del todo
                p['wall'] = np.full((5, 5), -1, dtype=int)
                p['score'] = 0
            p['floor_line'] = np.full(self.L_floor, -1, dtype=int)

        # Clear factories and center before refill
        self.factories[:] = 0
        self.center[:] = 0

        # Fill factories and center
        self.first_player_token = True
        self.first_player_next_round = -1
        self.current_player = 0
        self.round_count = 1
        self.done = False  # Reset done flag
        self._refill_factories()

        return self._get_obs()

    def step(self, action: Tuple[int, int, int], is_sim: bool = False):
        if len(self.get_valid_actions()) == 0:
            raise RuntimeError("No valid actions available. Possible deadlock.")
        source_idx, color, dest = action
        # Direct reference to player dict to ensure modifications persist
        p = self.players[self.current_player]
        
        before_score = p['score']
        
        # Helper to calculate current penalty BEFORE applying move
        current_penalty = calculate_floor_penalization(p['floor_line'])

        # Handle source removal
        if source_idx < self.N:
            # factory
            count = int(self.factories[source_idx, color])
            if self.factories[source_idx].sum() == 0 or count == 0:
                raise ValueError(f"Invalid action: factory {source_idx} has no tiles of color {color}")
            # move other colors to center
            other_colors = self.factories[source_idx].copy()
            other_colors[color] = 0
            self.center += other_colors
            # Ensure global state is mutated, avoid shallow copy issues
            for c in range(self.C):
                self.factories[source_idx, c] = 0
        elif source_idx == self.N:
            # center
            count = int(self.center[color])
            if self.center.sum() == 0 or count == 0:
                raise ValueError(f"Invalid action: center has no tiles of color {color}")
            self.center[color] = 0
            if self.first_player_token:
                # penalty token handling
                fl = p['floor_line']
                # Find first empty slot (-1)
                idxs = np.where(fl == -1)[0]
                if idxs.size > 0:
                    fl[idxs[0]] = 5 # Representing the -1 penalty token
                else:
                    # Fix Bug 5: If floor is full, it replaces the last tile
                    # This ensures the player holds the token AND pays the max penalty
                    fl[-1] = 5
                
                self.first_player_token = False
                self.first_player_next_round = self.current_player

        # Place tiles
        speculative_points = 0
        
        if dest < 5:
            # Check if line was already full (shouldn't be, valid_actions prevents it)
            # But we check if it BECOMES full
            was_full = all(slot != -1 for slot in p['pattern_lines'][dest])
            
            new_line, overflow = place_on_pattern_line(p['pattern_lines'][dest], color, count)
            p['pattern_lines'][dest] = new_line
            # Explicitly write back to main data structure in case of view/copy issues
            self.players[self.current_player]['pattern_lines'][dest] = new_line
            
            # Speculative Wall Points & Bonuses
            is_full = all(slot != -1 for slot in new_line)
            if is_full and not was_full:
                # Calculate what points we WOULD get
                # We use a clone of the wall to not affect game state
                temp_wall = p['wall'].copy()
                
                # 1. Placement Points
                points = transfer_to_wall(temp_wall, new_line.tolist(), dest)
                speculative_points += points
                
                # 2. Speculative Bonuses (Row/Col/Color)
                # Calculate bonus BEFORE this placement (should be based on real wall)
                current_bonus = calculate_final_bonus(p['wall'])
                # Calculate bonus AFTER this placement (on temp wall)
                new_bonus = calculate_final_bonus(temp_wall)
                bonus_delta = new_bonus - current_bonus
                
                speculative_points += bonus_delta
            
            # overflow to floor
            fl = p['floor_line']
            for _ in range(overflow):
                idxs = np.where(fl == -1)[0]
                if idxs.size > 0:
                    fl[idxs[0]] = color
                else:
                    # Fix Bug 4: Overflow goes to discard if floor is full
                    self.discard[color] += 1
        else:
            # all to floor
            fl = p['floor_line']
            for _ in range(count):
                idxs = np.where(fl == -1)[0]
                if idxs.size > 0:
                    fl[idxs[0]] = color
                else:
                    # Fix Bug 4: Overflow goes to discard if floor is full
                    self.discard[color] += 1

        # Calculate new penalty
        new_penalty = calculate_floor_penalization(p['floor_line'])
        penalty_delta = new_penalty - current_penalty
        
        # Apply speculative updates
        total_delta = penalty_delta + speculative_points
        p['score'] += total_delta
        self.round_accumulated_score[self.current_player] += total_delta

        # Check round end
        done = False
        reward = 0
        opponent = (self.current_player + 1) % self.num_players
        opponent_score_before = self.players[opponent]['score']
        if self._is_round_over():
            done = self._end_round()
            self.done = done
            if done:
                 # Reward = Delta Self - Delta Oppt (Full difference)
                 delta_self = p['score'] - before_score
                 delta_opp = self.players[opponent]['score'] - opponent_score_before
                 reward = delta_self - delta_opp
        else:
            # Next player turn
            self.current_player = opponent
        obs = self._get_obs()
        info = {'p0_score': p['score'], 'p1_score': self.players[opponent]['score'], 'round': self.round_count}
        return obs, reward, done, info

    def _refill_factories(self):
        # Empty center
        self.center[:] = 0
        # Fill each factory with 4 tiles
        for i in range(self.N):
            tiles = []
            for _ in range(4):
                # Refill bag from discard if bag is empty and discard has tiles
                if self.bag.sum() == 0 and self.discard.sum() > 0:
                    self.bag += self.discard
                    self.discard[:] = 0
                total = self.bag.sum()
                if total == 0:
                    # No tiles left anywhere, stop filling
                    break
                    
                probs = self.bag / total
                # choose a tile
                tile = np.random.choice(self.C, p=probs)
                tiles.append(tile)
                # decrement bag
                self.bag[tile] -= 1
            
            # place tiles into factory
            for t in tiles:
                self.factories[i, t] += 1

    def _is_round_over(self) -> bool:
        if any(self.factories[i].sum() > 0 for i in range(self.N)):
            return False
        if self.center.sum() > 0:
            return False
        return True

    def _end_round(self) -> bool:
        # 1. Revert Speculative Scoring
        for i, p in enumerate(self.players):
            p['score'] -= self.round_accumulated_score[i]
        
        # Reset accumulator for safety (though next round resets it too, good practice)
        self.round_accumulated_score = [0] * self.num_players

        # Score placement and penalties
        for p in self.players:
            # pattern lines -> wall
            for row_idx, line in enumerate(p['pattern_lines']):
                if -1 not in line:
                    color = int(line[0])
                    pts = transfer_to_wall(p['wall'], line, row_idx)
                    p['score'] += pts
                    # discard leftover tiles
                    leftover = len(line) - 1
                    self.discard[color] += leftover
                    p['pattern_lines'][row_idx][:] = -1
                    
            # floor line penalties
            pen = calculate_floor_penalization(p['floor_line'])
            p['score'] += pen
            for tile in p['floor_line']:
                if tile >= 0 and tile < 5: # 0-4 are colors
                    self.discard[int(tile)] += 1
                # tile 5 is the first player token, doesn't go to discard
            

            
            p['floor_line'][:] = -1
            

        # Check game end (any full wall row) OR max rounds reached
        # Check game end (any full wall row)
        game_over = any(all(cell != -1 for cell in row) for p in self.players for row in p['wall'])
        
        # NEW: Track termination reason
        if game_over:
            self.termination_reason = "normal_end"
        else:
            self.termination_reason = "normal_end"

        if game_over:
            # Apply final bonuses to each player
            for p in self.players:
                bonus = calculate_final_bonus(p['wall'])
                p['score'] += bonus
        else:
            self.first_player_token = True
            self._refill_factories()
            
            # Determine next starting player
            if self.first_player_next_round != -1:
                self.current_player = self.first_player_next_round
            else:
                # Should not happen in standard play if someone took from center
                # But if center was empty (rare?), keep current or random?
                # In Azul, center always has the -1 token initially.
                # So someone MUST have taken it.
                pass
            
            self.first_player_next_round = -1 # Reset for next round

        self.round_count += 1
        return game_over

    def _get_obs(self):
        return {
            'bag': self.bag.copy(),
            'discard': self.discard.copy(),
            'factories': self.factories.copy(),
            'center': self.center.copy(),
            'first_player_token': self.first_player_token,
            'players': [
                {
                    'pattern_lines': [np.array(line, dtype=int) for line in p['pattern_lines']],
                    'pattern_lines_padded': [np.pad(pl.copy(), (0, 5 - len(pl)), constant_values=-1) for pl in p['pattern_lines']],
                    'wall': p['wall'].copy(),
                    'floor_line': p['floor_line'].copy(),
                    'score': p['score']
                } for p in self.players
            ],
            'current_player': self.current_player,
            'round_count': self.round_count
        }

    def encode_observation(self, obs: dict) -> np.ndarray:
        """
        Encode the observation dict into a flat numpy array.
        Layout: [Spatial (20 channels * 5 * 5) | Factories | Global]
        
        Spatial Channels (20):
        - For each player (Current, then Opponent):
          - Pattern Lines (5 channels): Channel i = 1 where color i exists
          - Wall (5 channels): Channel i = 1 where color i exists
        
        Global features:
        - bag (5), discard (5), first_player_token (1)
        - floor_lines (num_players * 7), scores (num_players)
        - round_count (1, normalized)
        - bonuses per player: completed_rows, completed_cols, completed_colors (3 * num_players)
        - remaining_tiles (5)
        """
        # parts: bag, discard, factories, center
        # Spatial parts: pattern lines and walls
        spatial_parts = []
        # Canonicalize: Rotate players so current_player is at index 0
        current_player = obs['current_player']
        num_players = len(obs['players'])
        rotated_players = [obs['players'][(current_player + i) % num_players] for i in range(num_players)]

        # One-Hot Encoding for Spatial Features
        # Pattern Lines: 5 colors -> 5 planes per player
        for p in rotated_players:
            # Create 5 planes for pattern lines
            # Each plane (5x5) corresponds to a color 0..4
            pattern_planes = np.zeros((5, 5, 5), dtype=int)
            
            for row_idx, line in enumerate(p['pattern_lines']):
                # line is a numpy array of tile indices (or -1)
                for col_idx, tile in enumerate(line):
                    if tile != -1:
                        # Mark the cell (row_idx, col_idx) in specific color plane
                        pattern_planes[tile, row_idx, col_idx] = 1
            
            spatial_parts.append(pattern_planes.flatten())

        # Wall: 5 colors -> 5 planes per player
        for p in rotated_players:
            wall_planes = np.zeros((5, 5, 5), dtype=int)
            wall = p['wall']
            for r in range(5):
                for c in range(5):
                    tile = wall[r, c]
                    if tile != -1:
                        wall_planes[tile, r, c] = 1
            
            spatial_parts.append(wall_planes.flatten())
            
        # Factories parts: factories (N*5), center (5)
        factories_parts = [
            obs['factories'].flatten(),
            obs['center']
        ]
        
        # Global parts: bag, discard, first_player, floor_lines, scores
        
        # One-Hot Round Count (8 positions: 0=R1 ... 7=R8+)
        r_idx = min(obs['round_count'] - 1, 7)
        round_one_hot = np.zeros(8, dtype=np.float32)
        if r_idx >= 0:
            round_one_hot[r_idx] = 1.0

        global_parts = [
            obs['bag'],
            obs['discard'],
            np.array([int(obs['first_player_token'])], dtype=int),
            round_one_hot
        ]
        
        # floor_lines (rotated)
        floors = np.stack([p['floor_line'] for p in rotated_players])
        global_parts.append(floors.flatten())
        # scores (rotated)
        scores = np.array([p['score'] for p in rotated_players], dtype=int)
        global_parts.append(scores)
        
        # Bonuses (completed rows, columns, colors per player)
        for p in rotated_players:
            wall = p['wall']
            # Completed rows
            completed_rows = sum(1 for row in wall if all(cell != -1 for cell in row))
            # Completed columns
            completed_cols = sum(1 for col_idx in range(5) if all(wall[row_idx][col_idx] != -1 for row_idx in range(5)))
            # Completed colors
            color_counts = np.zeros(5, dtype=int)
            for row in wall:
                for cell in row:
                    if cell != -1:
                        color_counts[cell] += 1
            completed_colors = sum(1 for count in color_counts if count == 5)
            
            global_parts.append(np.array([completed_rows, completed_cols, completed_colors], dtype=int))
        
        # Remaining tiles
        visible_factories = obs['factories'].sum(axis=0) 
        visible_center = obs['center']
        remaining = obs['bag'] + obs['discard'] + visible_factories + visible_center
        global_parts.append(remaining)
        
        # Concatenate
        spatial_flat = np.concatenate(spatial_parts)
        factories_flat = np.concatenate(factories_parts)
        global_flat = np.concatenate(global_parts)
        
        return np.concatenate([spatial_flat, factories_flat, global_flat])

    def action_to_index(self, action: Tuple[int, int, int]) -> int:
        """
        Convert an action tuple (source_idx, color, dest) into a flat index.
        """
        source_idx, color, dest = action
        return source_idx * (self.C * 6) + color * 6 + dest

    def clone(self) -> 'AzulEnv':
        # Optimized clone without deepcopy for speed
        new = AzulEnv.__new__(AzulEnv)
        gym.Env.__init__(new)
        
        # Copy scalars
        new.num_players = self.num_players
        new.C = self.C
        new.N = self.N
        new.max_rounds = self.max_rounds
        new.L_floor = self.L_floor
        new.action_space = self.action_space
        new.observation_space = self.observation_space
        new.action_size = self.action_size

        new.round_count = self.round_count
        new.done = self.done
        new.first_player_token = self.first_player_token
        new.first_player_next_round = self.first_player_next_round
        new.current_player = self.current_player
        new.termination_reason = getattr(self, 'termination_reason', 'normal_end')
        new.round_accumulated_score = self.round_accumulated_score[:]
        
        # Copy numpy arrays (fast)
        new.bag = self.bag.copy()
        new.discard = self.discard.copy()
        new.factories = self.factories.copy()
        new.center = self.center.copy()
        
        # Manually copy players list of dicts
        # This is much faster than deepcopying the whole list
        new.players = []
        for p in self.players:
            new_p = {
                'pattern_lines': [line.copy() for line in p['pattern_lines']],
                'wall': p['wall'].copy(),
                'floor_line': p['floor_line'].copy(),
                'score': p['score']
            }
            new.players.append(new_p)

        return new

    def index_to_action(self, index: int) -> Tuple[int, int, int]:
        """
        Convert a flat index into an action tuple (source_idx, color, dest).
        """
        dest = index % 6
        color = (index // 6) % self.C
        source_idx = index // (6 * self.C)
        return (source_idx, color, dest)

    def render(self, mode='human'):
        print(f"Player to move: {self.current_player}")
        for idx, p in enumerate(self.players):
            print(f"== Player {idx} ==")
            print("Score:", p['score'])
            print("Wall:\n", p['wall'])
            print("Pattern lines:")
            for line in p['pattern_lines']:
                print(" ", line)
            print("Floor line:", p['floor_line'])
        print("Factories:\n", self.factories)
        print("Center:", self.center, "First token present:", self.first_player_token)

    def get_valid_actions(self) -> list:
        """
        Returns a list of valid actions (source_idx, color, dest).
        An action is valid if the source has at least one tile of the chosen color,
        passes validate_origin, and no conflict with wall rules.
        """
        valid_actions = []
        for source_idx in range(self.N + 1):  # factories and center
            source = self.factories[source_idx] if source_idx < self.N else self.center
            origin = ("factory", source_idx) if source_idx < self.N else ("center", None)
            for color in range(self.C):
                if source[color] == 0:
                    continue
                if not validate_origin(self.factories, self.center, origin, color):
                    continue
                for dest in range(6):
                    if dest < 5:
                        wall_row = self.players[self.current_player]['wall'][dest]
                        if color in wall_row:
                            continue  # ya está ese color en la fila del muro
                        
                        # Fix Bug 3: Check if pattern line has a different color
                        pattern_line = self.players[self.current_player]['pattern_lines'][dest]
                        # If line is not empty (has -1s but first element is not -1)
                        # In this implementation, we fill from left? No, we fill indices.
                        # But place_on_pattern_line checks `any(slot != -1 and slot != color ...)`
                        # Here we just need to check if there is ANY slot with a different color.
                        # Since a line can only hold ONE color (plus empty slots), we can just check the first non-empty slot?
                        # Or just check if any slot is != -1 and != color.
                        if any(slot != -1 and slot != color for slot in pattern_line):
                            continue

                    valid_actions.append((source_idx, color, dest))
        return valid_actions
    
    def get_action_mask(self) -> np.ndarray:
        """
        Returns a binary mask of valid actions (1=legal, 0=illegal).
        """
        mask = np.zeros(self.action_size, dtype=np.float32)
        valid_actions = self.get_valid_actions()
        for action in valid_actions:
            idx = self.action_to_index(action)
            mask[idx] = 1.0
        return mask

    def get_debug_wall_value(self, player_idx:int) -> int:
        """
        Recorre el muro del jugador sumando 1 si la celda es distinta de -1 y 0 en caso contrario.
        Devuelve el valor total.
        """

        wall = self.players[player_idx]['wall']
        total = 0
        for row in wall:
            for cell in row:
                if cell != -1:
                    total += 1
        return total
    
    def get_final_scores(self):
        return [p['score'] for p in self.players]
