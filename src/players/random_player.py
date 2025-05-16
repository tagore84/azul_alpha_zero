# src/players/random_player.py
import random
from .base_player import BasePlayer

class RandomPlayer(BasePlayer):
    """
    Jugador que elige entre todas las acciones legales de forma uniforme al azar.
    """

    def __init__(self):
        super().__init__()

    def predict(self, obs):
        """
        Dada una observación obs, construye la lista de acciones legales
        y devuelve una elegida al azar.
        """
        C = 5  # número de colores
        D = 6  # destinos: 5 filas de patrón (0–4) + 1 línea de suelo (5)
        num_factories = len(obs["factories"])
        num_sources = num_factories + 1  # fábricas + centro
        current = obs["current_player"]

        legal_actions = []

        for src in range(num_sources):
            # tile_counts en fábrica o en el centro
            tile_counts = (
                obs["factories"][src]
                if src < num_factories
                else obs["center"]
            )

            for color in range(C):
                if tile_counts[color] == 0:
                    continue  # no hay fichas de ese color en la fuente

                for dest in range(D):
                    index = src * (C * D) + color * D + dest

                    if dest < C:
                        # destino es fila de patrón dest (0–4)
                        line = obs["players"][current]["pattern_lines"][dest]
                        # legal si hay al menos una plaza libre (0) y ningún otro color distinto
                        has_space = any(tile == 0 for tile in line)
                        same_color_ok = all(tile in (0, color) for tile in line)
                        if has_space and same_color_ok:
                            legal_actions.append(index)
                    else:
                        # dest == 5: siempre legal mandar al suelo
                        legal_actions.append(index)

        if not legal_actions:
            raise RuntimeError("No hay acciones legales disponibles")

        return random.choice(legal_actions)