# src/players/lillo_expertillo.py

from enum import Enum
from .base_player import BasePlayer

C = 5  # número de colores
D = 6  # destinos: 5 filas de patrón (0–4) + 1 línea de suelo (5)

class OptionType(Enum):
    TWO_STEPS_COMPLETE = 0
    ONE_STEP_COMPLETE  = 1
    INCOMPLETE         = 2
    ALL_TO_FLOOR       = 3

class Option:
    """
    Representa una posible acción: tomar `num` fichas de color `color`
    de la fuente `src` y colocarlas en la fila `row` (0–4 patrón, 5 suelo).
    """
    def __init__(self, src, color, row, num):
        self.src    = src
        self.color  = color
        self.row    = row
        self.num    = num
        # se fijarán en evaluate()
        self.type     = None
        self.slotNum  = 0
        self.floorNum = 0

    def evaluate(self, obs):
        """
        Asigna self.type, self.slotNum y self.floorNum según el estado obs.
        """
        current = obs["current_player"]
        # obtener la fila de destino
        if self.row < C:
            slot_row = obs["players"][current]["pattern_lines"][self.row]
        else:
            slot_row = obs["players"][current]["floor_line"]

        empty_slots  = sum(1 for t in slot_row if t == 0)
        filled_slots = len(slot_row) - empty_slots

        complete            = (empty_slots <= self.num)
        two_steps_complete  = complete and (filled_slots > 0)
        all_to_floor        = (empty_slots == 0)

        if self.row < C:
            if all_to_floor:
                self.type = OptionType.ALL_TO_FLOOR
            elif two_steps_complete:
                self.type = OptionType.TWO_STEPS_COMPLETE
            elif complete:
                self.type = OptionType.ONE_STEP_COMPLETE
            else:
                self.type = OptionType.INCOMPLETE

            self.slotNum  = min(empty_slots, self.num)
            self.floorNum = max(0, self.num - empty_slots)
        else:
            # siempre legal mandar todo a suelo
            self.type     = OptionType.ALL_TO_FLOOR
            self.slotNum  = 0
            self.floorNum = self.num

    def __lt__(self, other):
        # Primero por tipo (ordinal menor = mejor)
        if self.type != other.type:
            return self.type.value < other.type.value

        # Dentro del mismo tipo:
        if self.type in (
            OptionType.TWO_STEPS_COMPLETE,
            OptionType.ONE_STEP_COMPLETE,
            OptionType.INCOMPLETE
        ):
            # queremos slotNum mayor primero
            return self.slotNum  > other.slotNum
        else:
            # ALL_TO_FLOOR: queremos num mayor primero
            return self.num       > other.num

    def __eq__(self, other):
        return (
            self.type     == other.type
            and self.slotNum  == other.slotNum
            and self.floorNum == other.floorNum
            and self.src      == other.src
            and self.color    == other.color
            and self.row      == other.row
            and self.num      == other.num
        )

class LilloExpertillo(BasePlayer):
    """
    Traducción de LilloExpertillo (Java) a Python.
    Juega escogiendo la opción evaluada como “mejor” según OptionType y recuentos.
    """
    def __init__(self):
        super().__init__()

    def predict(self, obs):
        """
        Construye todas las opciones legales, las evalúa, ordena
        y devuelve el índice entero de la mejor.
        """
        num_factories = len(obs["factories"])
        num_sources   = num_factories + 1  # fábricas + centro
        options       = []

        # generar todas las opciones
        for src in range(num_sources):
            tile_counts = (
                obs["factories"][src]
                if src < num_factories
                else obs["center"]
            )
            for color in range(C):
                count = tile_counts[color]
                if count == 0:
                    continue
                for row in range(D):
                    opt = Option(src, color, row, count)
                    opt.evaluate(obs)
                    # si type no es None, es legal
                    if opt.type is not None:
                        options.append(opt)

        if not options:
            raise RuntimeError("No hay opciones legales (LilloExpertillo)")

        # ordenar y quedarnos con la mejor (0 tras sort)
        options.sort()
        best = options[0]

        # codificar acción en un índice entero
        index = best.src * (C * D) + best.color * D + best.row
        return index