# src/azul/env.py

import gym
from gym import spaces
import numpy as np
from typing import Tuple

from azul.utils import print_wall
from .rules import validate_origin, place_on_pattern_line, transfer_to_wall, calculate_round_score, calculate_final_bonus, Color
import random  # Añade esto al principio del archivo
from constants import SEED
import copy  # Add this import at the top of the file if not present

class AzulEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, num_players: int = 2, factories_count: int = 5, seed: int = None):
        super().__init__()
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
        self.num_players = num_players
        self.C: int = len(Color)
        self.N: int = factories_count
        self.L_floor: int = 7

        # Game state
        self.bag: np.ndarray = np.full(self.C, 20, dtype=int)
        self.discard: np.ndarray = np.zeros(self.C, dtype=int)
        self.factories: np.ndarray = np.zeros((self.N, self.C), dtype=int)
        self.center: np.ndarray = np.zeros(self.C, dtype=int)
        self.first_player_token: bool = False

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
        else:
            # center
            count = int(self.center[color])
            if self.center.sum() == 0 or count == 0:
                raise ValueError(f"Invalid action: center has no tiles of color {color}")
            self.center[color] = 0
            if self.first_player_token:
                # penalty token
                fl = p['floor_line']
                free = np.where(fl == 0)[0]
                if free.size > 0:
                    fl[free[0]] = -1
                self.first_player_token = False

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
        info = {}
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
            pen = calculate_round_score(p['wall'], p['floor_line'])
            p['score'] += pen
            for tile in p['floor_line']:
                if tile > 0:
                    self.discard[int(tile)-1] += 1
            p['floor_line'] = np.full(self.L_floor, -1, dtype=int)
            

        # Check game end (any full wall row)
        game_over = any(all(cell != -1 for cell in row) for p in self.players for row in p['wall'])
        if game_over:
            # Apply final bonuses to each player
            for p in self.players:
                bonus = calculate_final_bonus(p['wall'])
                p['score'] += bonus
        else:
            self.first_player_token = True
            self._refill_factories()
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
        parts = [
            obs['bag'],
            obs['discard'],
            obs['factories'].flatten(),
            obs['center'],
            np.array([int(obs['first_player_token'])], dtype=int)
        ]
        # players pattern_lines padded to 5x5
        pattern = []
        for p in obs['players']:
            plines = np.full((5, 5), -1, dtype=int)
            for i, line in enumerate(p['pattern_lines']):
                plines[i, :len(line)] = line
            pattern.append(plines)
        parts.append(np.array(pattern).flatten())
        # walls
        walls = np.stack([p['wall'] for p in obs['players']])
        parts.append(walls.flatten())
        # floor_lines
        floors = np.stack([p['floor_line'] for p in obs['players']])
        parts.append(floors.flatten())
        # scores
        scores = np.array([p['score'] for p in obs['players']], dtype=int)
        parts.append(scores)
        # current player
        parts.append(np.array([obs['current_player']], dtype=int))
        # concatenate and return
        return np.concatenate(parts)

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

        # Copia de estado del juego
        new.bag = self.bag.copy()
        new.discard = self.discard.copy()
        new.factories = self.factories.copy()
        new.center = self.center.copy()
        new.first_player_token = self.first_player_token
        new.current_player = self.current_player
        new.players = copy.deepcopy(self.players)
        new.round_count = self.round_count

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

