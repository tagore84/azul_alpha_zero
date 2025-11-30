
import numpy as np
import random
from .base_player import BasePlayer

class RandomPlusPlayer(BasePlayer):
    def __init__(self, name="RandomPlus"):
        super().__init__()
        self.name = name

    def predict(self, obs: dict) -> tuple:
        """
        Elegir una acción aleatoria entre las que envían menos azulejos al suelo.
        """
        # Necesitamos acceso al env para simular, pero obs solo tiene el estado.
        # En este framework, los players suelen recibir solo obs.
        # Sin embargo, para calcular esto necesitamos saber cuántas fichas hay en cada fábrica/centro.
        # La obs tiene 'factories', 'center', 'players'.
        
        valid_actions = self.get_valid_actions(obs)
        if not valid_actions:
            return None # Should not happen

        min_floor_tiles = float('inf')
        best_actions = []

        current_player_idx = obs['current_player']
        p_board = obs['players'][current_player_idx]
        pattern_lines = p_board['pattern_lines']
        # floor_line = p_board['floor_line'] # No necesitamos saber qué hay, solo cuánto añadimos
        
        # Check if first player token is in center
        first_player_token_available = (obs['center'][-1] == 1) # Asumiendo que el token es el último o se maneja aparte.
        # Revisando env.py, center es un array de counts. El token suele ser un flag en env, pero en obs?
        # En obs['center'] suele ser un array de counts por color.
        # El token de primer jugador se maneja a veces como un color extra o un flag.
        # Mirando env.py: self.center = np.zeros(self.C, dtype=int). No parece tener el token.
        # El token está en self.first_player_token (bool).
        # ¿Está en obs?
        # obs = {'factories': ..., 'center': ..., 'players': ..., 'current_player': ..., 'round': ...}
        # Si no está en obs, no podemos saberlo con certeza solo con obs estándar.
        # Pero asumamos que si tomamos del centro y hay fichas, y nadie lo ha tomado...
        # Espera, si 'center' tiene fichas, alguien tiene que tomar el -1 si es el primero.
        # ¿Cómo sabemos si ya se tomó?
        # Si es el primer turno que se toma del centro.
        # Podemos inferirlo: si obs['center'] tiene fichas y es la primera vez que alguien toma de ahí... difícil sin historia.
        # PERO, en muchos envs de Azul, el token se representa.
        # Vamos a asumir por ahora que ignoramos el token -1 para simplificar, o miramos si podemos deducirlo.
        # O mejor, miramos si el env pasa el objeto env al player (algunos frameworks lo hacen).
        # En tournament.py: action = current.predict(obs). Solo obs.
        
        # Vamos a implementar la lógica de conteo de fichas.
        
        for action in valid_actions:
            source_idx, color, dest = action
            
            # 1. Cuántas fichas tomamos?
            count = 0
            # Note: Center is source_idx = N (not -1)
            if source_idx < len(obs['factories']):
                # Factory
                count = obs['factories'][source_idx][color]
            else:
                # Center (source_idx == N)
                count = obs['center'][color]
            
            # 2. Cuántas van al suelo?
            floor_count = 0
            
            # Note: Floor is dest=5, not dest=-1
            if dest == 5:
                # Todas al suelo
                floor_count += count
            else:
                # Pattern line
                # Capacidad de la línea: dest + 1 (fila 0 tiene cap 1, fila 4 tiene cap 5)
                capacity = dest + 1
                
                # Cuántas hay ya?
                # pattern_lines[dest] es un array. -1 es vacío.
                # Contamos los que no son -1.
                current_filled = np.sum(pattern_lines[dest] != -1)
                
                space = capacity - current_filled
                
                if count > space:
                    floor_count += (count - space)
            
            if floor_count < min_floor_tiles:
                min_floor_tiles = floor_count
                best_actions = [action]
            elif floor_count == min_floor_tiles:
                best_actions.append(action)
        
        return random.choice(best_actions)

    def get_valid_actions(self, obs):
        # IMPORTANT: Make copy to prevent mutation issues
        factories = np.array(obs['factories'], copy=True)
        center = np.array(obs['center'], copy=True)
        p_board = obs['players'][obs['current_player']]
        pattern_lines = p_board['pattern_lines']
        wall = p_board['wall']
        
        valid = []
        
        # Sources: factories (0..N-1) and center (-1)
        # Colors: 0..C-1
        # Dests: 0..4 and -1 (floor)
        
        # Check factories
        for i, factory in enumerate(factories):
            for color, count in enumerate(factory):
                if count > 0:
                    # Valid source/color. Check dests.
                    for dest in range(5):
                        if self._is_legal_move(pattern_lines, wall, dest, color):
                            valid.append((i, color, dest))
                    # Floor always valid (dest=5)
                    valid.append((i, color, 5))
        
        # Check center (source_idx = N = len(factories))
        N = len(factories)
        for color in range(len(center)):
            count = center[color]
            if count > 0:
                for dest in range(5):
                    if self._is_legal_move(pattern_lines, wall, dest, color):
                        valid.append((N, color, dest))
                valid.append((N, color, 5))
                
        return valid

    def _is_legal_move(self, pattern_lines, wall, dest, color):
        # 1. Pattern line not full
        if -1 not in pattern_lines[dest]:
            return False
        
        # 2. Pattern line empty or same color
        first_val = pattern_lines[dest][0]
        if first_val != -1 and first_val != color:
            return False
            
        # 3. Wall row doesn't have this color
        # wall[dest] is the row, we need to check if color appears in it
        if color in wall[dest]:
            return False
            
        return True
