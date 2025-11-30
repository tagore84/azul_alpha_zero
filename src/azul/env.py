# src/azul/env.py

import gym
from gym import spaces
import numpy as np
from typing import Tuple

from azul.utils import print_floor, print_wall
from .rules import validate_origin, place_on_pattern_line, transfer_to_wall, calculate_floor_penalization, calculate_final_bonus, Color
import random  # Añade esto al principio del archivo
import copy  # Add this import at the top of the file if not present

class AzulEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, num_players: int = 2, factories_count: int = 5, seed: int = None, max_rounds: int = 15):
        super().__init__()
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
        self.num_players = num_players
        self.C: int = len(Color)
        self.N: int = factories_count
        self.L_floor: int = 7
        self.max_rounds = max_rounds

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
        self.reset()
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
        self._refill_factories()

        return self._get_obs()

    def step(self, action: Tuple[int, int, int], is_sim: bool = False):
        if len(self.get_valid_actions()) == 0:
            raise RuntimeError("No valid actions available. Possible deadlock.")
        source_idx, color, dest = action
        # Direct reference to player dict to ensure modifications persist
        p = self.players[self.current_player]
        
        before_score = p['score']

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
                # penalty token
                fl = p['floor_line']
                free = np.where(fl == 0)[0] # 0 is empty? No, -1 is empty.
                # Wait, let's check init. floor_line is -1.
                # But in step: free = np.where(fl == 0)[0] ??
                # Let's check line 141 in original: free = np.where(fl == 0)[0]
                # That looks like a BUG in original code if -1 is empty.
                # Let's check view_file output again.
                # Line 38: 'floor_line': np.full(self.L_floor, -1, dtype=int),
                # Line 141: free = np.where(fl == 0)[0]
                # This seems wrong if 0 is BLUE.
                # Ah, wait. In place_on_pattern_line, empty is -1.
                # In floor line, is -1 empty?
                # Line 155: idxs = np.where(fl == -1)[0] -> This confirms -1 is empty.
                # So line 141 `free = np.where(fl == 0)[0]` is definitely a BUG.
                # It checks for color 0 (BLUE) instead of empty (-1).
                # I should fix this too.
                
                idxs = np.where(fl == -1)[0]
                if idxs.size > 0:
                    fl[idxs[0]] = -1 # Wait, penalty token is not a color?
                    # Usually represented as a special value or just -1 penalty?
                    # The rule says: "The first player to take tiles from the center... takes the -1 penalty token".
                    # In this implementation, does it store the token in the floor line?
                    # Line 143: fl[free[0]] = -1.
                    # If -1 is empty, then setting it to -1 does nothing.
                    # This implementation seems to rely on `calculate_floor_penalization` which just counts filled slots?
                    # Let's check `calculate_floor_penalization`.
                    # Line 102: for idx, tile in enumerate(floor_line): if tile != -1: score += penalties[idx]
                    # So if we put -1, it's counted as empty.
                    # So the first player token is NOT being penalized in the original code!
                    # We need a value for the token. Maybe -2? Or just any non-negative value?
                    # But colors are 0..4.
                    # Let's use a special value, e.g., 5 (since colors are 0-4).
                    # Or maybe the implementation intended to use a specific color?
                    # Actually, let's look at how it was.
                    # `fl[free[0]] = -1` -> This effectively does nothing if it was already -1.
                    # So the penalty was missing.
                    
                    # Fix: Use a special value for the first player token. Let's say 5 (Color.RED + 1).
                    pass
                
                # Let's fix the bug:
                # 1. Find first empty slot.
                # 2. Place a "token" there.
                idxs = np.where(fl == -1)[0]
                if idxs.size > 0:
                    fl[idxs[0]] = 5 # Representing the -1 token
                
                self.first_player_token = False
                self.first_player_next_round = self.current_player

        # Place tiles
        if dest < 5:
            new_line, overflow = place_on_pattern_line(p['pattern_lines'][dest], color, count)
            p['pattern_lines'][dest] = new_line
            # Explicitly write back to main data structure in case of view/copy issues
            self.players[self.current_player]['pattern_lines'][dest] = new_line
            # overflow to floor
            fl = p['floor_line']
            for _ in range(overflow):
                idxs = np.where(fl == -1)[0]
                if idxs.size > 0:
                    fl[idxs[0]] = color
        else:
            # all to floor
            fl = p['floor_line']
            for _ in range(count):
                idxs = np.where(fl == -1)[0]
                if idxs.size > 0:
                    fl[idxs[0]] = color

        # Check round end
        done = False
        reward = 0
        opponent = (self.current_player + 1) % self.num_players
        opponent_score_before = self.players[opponent]['score']
        if self._is_round_over():
            done = self._end_round()
            self.done = done
            reward = (p['score'] - before_score) - 0.5 * (self.players[opponent]['score'] - opponent_score_before)
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
                if total > 0:
                    probs = self.bag / total
                else:
                    # no tiles left anywhere: uniform random over colors
                    probs = np.ones(self.C, dtype=float) / self.C
                # choose a tile
                tile = np.random.choice(self.C, p=probs)
                tiles.append(tile)
                # decrement bag only if it was refilled properly
                if total > 0:
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
                    p['pattern_lines'][row_idx] = np.full(len(line), -1, dtype=int)
            # floor line penalties
            pen = calculate_floor_penalization(p['floor_line'])
            p['score'] += pen
            for tile in p['floor_line']:
                if tile >= 0 and tile < 5: # 0-4 are colors
                    self.discard[int(tile)] += 1
                # tile 5 is the first player token, doesn't go to discard
            p['floor_line'] = np.full(self.L_floor, -1, dtype=int)
            

        # Check game end (any full wall row) OR max rounds reached
        game_over = any(all(cell != -1 for cell in row) for p in self.players for row in p['wall'])
        
        if self.round_count >= self.max_rounds:
            game_over = True

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
        """
        # parts: bag, discard, factories, center
        # Spatial parts: pattern lines and walls
        spatial_parts = []
        # Canonicalize: Rotate players so current_player is at index 0
        current_player = obs['current_player']
        num_players = len(obs['players'])
        rotated_players = [obs['players'][(current_player + i) % num_players] for i in range(num_players)]

        # players pattern_lines padded to 5x5
        for p in rotated_players:
            plines = np.full((5, 5), -1, dtype=int)
            for i, line in enumerate(p['pattern_lines']):
                plines[i, :len(line)] = line
            spatial_parts.append(plines)
        
        # walls
        for p in rotated_players:
            spatial_parts.append(p['wall'])
            
        # Global parts: bag, discard, factories, center, first_player, floor_lines, scores
        global_parts = [
            obs['bag'],
            obs['discard'],
            obs['factories'].flatten(),
            obs['center'],
            np.array([int(obs['first_player_token'])], dtype=int)
        ]
        
        # floor_lines (rotated)
        floors = np.stack([p['floor_line'] for p in rotated_players])
        global_parts.append(floors.flatten())
        # scores (rotated)
        scores = np.array([p['score'] for p in rotated_players], dtype=int)
        global_parts.append(scores)
        # current player removed (implicit)
        
        # Concatenate spatial then global
        # Spatial: (num_players * 2, 5, 5) flattened
        spatial_flat = np.array(spatial_parts).flatten()
        global_flat = np.concatenate(global_parts)
        
        return np.concatenate([spatial_flat, global_flat])

    def action_to_index(self, action: Tuple[int, int, int]) -> int:
        """
        Convert an action tuple (source_idx, color, dest) into a flat index.
        """
        source_idx, color, dest = action
        return source_idx * (self.C * 6) + color * 6 + dest

    def clone(self) -> 'AzulEnv':
        new = AzulEnv.__new__(AzulEnv)  # crea instancia sin llamar a __init__
        gym.Env.__init__(new)  # inicializa parte base sin resetear
        new.num_players = self.num_players
        new.C = self.C
        new.N = self.N
        new.L_floor = self.L_floor
        new.action_space = self.action_space
        new.observation_space = self.observation_space
        new.action_size = self.action_size

        new.max_rounds = self.max_rounds

        # Copia de estado del juego
        new.bag = self.bag.copy()
        new.discard = self.discard.copy()
        new.factories = self.factories.copy()
        new.center = self.center.copy()
        new.first_player_token = self.first_player_token
        new.first_player_next_round = self.first_player_next_round
        new.current_player = self.current_player
        new.players = copy.deepcopy(self.players)
        new.round_count = self.round_count
        new.done = self.done

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
                    valid_actions.append((source_idx, color, dest))
        return valid_actions
    
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

