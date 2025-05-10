# src/azul/rules.py

from enum import IntEnum
from typing import List, Tuple

class Color(IntEnum):
    BLUE = 0
    YELLOW = 1
    RED = 2
    BLACK = 3
    ORANGE = 4

def validate_origin(factories: List[List[int]], center: List[int], source: Tuple[str, int], color: Color) -> bool:
    """
    Checks that there is at least one tile of the chosen color
    in the specified factory or in the center.
    source = ("factory", idx) or ("center", None)
    """
    source_type, idx = source
    if source_type == "factory":
        return factories[idx][color] > 0
    elif source_type == "center":
        return center[color] > 0
    else:
        raise ValueError(f"Unknown source type: {source_type}")

def place_on_pattern_line(pattern_line: List[int], color: Color, count: int) -> Tuple[List[int], int]:
    """
    Attempts to place `count` tiles of `color` on a pattern line.
    Returns (new_line, overflow_to_floor).
    """
    # If line contains a different color, all tiles overflow
    if any(slot != -1 and slot != color for slot in pattern_line):
        return pattern_line.copy(), count

    empty_indices = [i for i, slot in enumerate(pattern_line) if slot == -1]
    placeable = min(len(empty_indices), count)
    new_line = pattern_line.copy()
    for i in range(placeable):
        new_line[empty_indices[i]] = color
    overflow = count - placeable
    return new_line, overflow

def transfer_to_wall(wall: List[List[int]], pattern_line: List[int], row: int) -> int:
    """
    Transfers the completed tile from the pattern line to the wall
    at the correct position and returns the points gained.
    """
    # Determine column for this color on the wall pattern
    color = pattern_line[0]  # all slots are same color when complete
    # Mapping of each row's color pattern
    row_patterns = [
        [Color.BLUE, Color.YELLOW, Color.ORANGE, Color.BLACK, Color.RED],
        [Color.RED, Color.BLUE, Color.YELLOW, Color.ORANGE, Color.BLACK],
        [Color.BLACK, Color.RED, Color.BLUE, Color.YELLOW, Color.ORANGE],
        [Color.ORANGE, Color.BLACK, Color.RED, Color.BLUE, Color.YELLOW],
        [Color.YELLOW, Color.ORANGE, Color.BLACK, Color.RED, Color.BLUE],
    ]
    col = row_patterns[row].index(color)
    wall[row][col] = color

    # Count contiguous tiles horizontally
    score = 1
    # left
    c = col - 1
    while c >= 0 and wall[row][c] != -1:
        score += 1
        c -= 1
    # right
    c = col + 1
    while c < 5 and wall[row][c] != -1:
        score += 1
        c += 1

    # Count contiguous tiles vertically
    v_count = 0
    # up
    r = row - 1
    while r >= 0 and wall[r][col] != -1:
        v_count += 1
        r -= 1
    # down
    r = row + 1
    while r < 5 and wall[r][col] != -1:
        v_count += 1
        r += 1

    if v_count > 0:
        score += v_count
    return score

def calculate_round_score(wall: List[List[int]], floor_line: List[int]) -> int:
    """
    Calculates the score for the round by evaluating the wall
    and the floor line according to the game rules.
    """
    # Penalties for each floor line position
    penalties = [-1, -1, -2, -2, -2, -3, -3]
    score = 0
    # Subtract penalties for each placed tile in the floor line
    for idx, tile in enumerate(floor_line):
        if tile != 0:  # non-zero indicates a tile is present
            score += penalties[idx]
    return score


def calculate_final_bonus(wall: List[List[int]]) -> int:
    """
    Calculates final end-of-game bonuses:
    +2 points for each complete row,
    +7 points for each complete column,
    +10 points for each complete set of one color.
    """
    bonus = 0
    # Complete rows
    for row in wall:
        if all(cell != -1 for cell in row):
            bonus += 2
    # Complete columns
    for col in range(len(wall[0])):
        if all(wall[row_idx][col] != -1 for row_idx in range(len(wall))):
            bonus += 7
    # Complete color sets
    for color in Color:
        count = sum(1 for row in wall for cell in row if cell == color)
        if count == len(wall):
            bonus += 10
    return bonus
