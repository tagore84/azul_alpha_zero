#!/usr/bin/env python3
"""
Script para verificar los cálculos de scoring en la transición Round 3 -> Round 4
"""

import numpy as np

# Patrón de colores del muro según rules.py
row_patterns = [
    [0, 1, 2, 3, 4],  # BLUE, YELLOW, ORANGE, BLACK, RED
    [4, 0, 1, 2, 3],  # RED, BLUE, YELLOW, ORANGE, BLACK
    [3, 4, 0, 1, 2],  # BLACK, RED, BLUE, YELLOW, ORANGE
    [2, 3, 4, 0, 1],  # ORANGE, BLACK, RED, BLUE, YELLOW
    [1, 2, 3, 4, 0],  # YELLOW, ORANGE, BLACK, RED, BLUE
]

def transfer_to_wall_scoring(wall, row, color, debug=False):
    """Calcula puntos según la lógica ACTUAL (con bug) de transfer_to_wall"""
    col = row_patterns[row].index(color)
    
    # Primero coloca la ficha (como hace el código real)
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
    
    if debug:
        print(f"       Debug: horizontal score = {score}, v_count = {v_count}")
    
    if v_count > 0:
        score += (v_count + 1)  # BUG: Suma la ficha dos veces
    
    return score, col


def calculate_floor_penalization(floor_line):
    """Calcula penalizaciones del floor line"""
    penalties = [-1, -1, -2, -2, -2, -3, -3]
    score = 0
    for idx, tile in enumerate(floor_line):
        if tile != -1:
            score += penalties[idx]
    return score

# ============================================
# PLAYER 1 - Round 3 -> Round 4
# ============================================
print("=" * 60)
print("PLAYER 1 - Round 3 -> Round 4")
print("=" * 60)

# Wall ANTES del end_round
wall_p1_before = np.array([
    [ 0,  1, -1, -1,  4],
    [-1,  0,  1, -1, -1],
    [ 3, -1, -1, -1, -1],
    [-1, -1, -1, -1, -1],
    [-1, -1, -1, -1, -1]
], dtype=int)

print("\nWall ANTES:")
print(wall_p1_before)

# Pattern lines completadas
print("\nPattern Lines completadas:")
print("  Row 2: [0 0 0]    (color 0)")
print("  Row 3: [3 3 3 3]  (color 3)")

# Floor line
floor_p1 = np.array([3, 3, 3, 0, -1, -1, -1])
print("\nFloor Line:", floor_p1)

# Simular colocación en muro - usamos una copia para cada vez
wall_p1_working = wall_p1_before.copy()

# Row 2, Color 0
pts_row2, col_row2 = transfer_to_wall_scoring(wall_p1_working, 2, 0, debug=True)
print(f"\n1. Colocar color 0 en row 2:")
print(f"   Columna: {col_row2}") 
print(f"   Puntos: {pts_row2}")
print(f"   Wall después de row 2:")
print(wall_p1_working)

# Row 3, Color 3
print(f"\n2. Preparando para colocar color 3 en row 3:")
print(f"   Wall ANTES de colocar row 3:")
print(wall_p1_working)

pts_row3, col_row3 = transfer_to_wall_scoring(wall_p1_working, 3, 3, debug=True)
print(f"\n   Columna donde va color 3: {col_row3}")
print(f"   Puntos calculados: {pts_row3}")
print(f"   Wall después:")
print(wall_p1_working)

# Verificar manualmente las verticales en col 1
print(f"\n   Verificación manual de vertical en col {col_row3}:")
for r in range(5):
    print(f"     Row {r}, Col {col_row3}: {wall_p1_working[r][col_row3]}")



# Penalizaciones
floor_penalty = calculate_floor_penalization(floor_p1)
print(f"\n3. Penalizaciones Floor Line: {floor_penalty}")

# Total
pts_wall = pts_row2 + pts_row3
pts_total = pts_wall + floor_penalty
score_before = -19
score_after = score_before + pts_total

print(f"\n{'='*40}")
print(f"RESUMEN PLAYER 1:")
print(f"{'='*40}")
print(f"Score antes:        {score_before}")
print(f"Puntos muro:        +{pts_wall} (row2:{pts_row2} + row3:{pts_row3})")
print(f"Penalizaciones:     {floor_penalty}")
print(f"CAMBIO NETO:        {pts_total}")
print(f"Score esperado:     {score_after}")
print(f"Score real (log):   -28")
print(f"DIFERENCIA:         {-28 - score_after}")

# ============================================
# PLAYER 0 - Round 3 -> Round 4
# ============================================
print("\n\n" + "=" * 60)
print("PLAYER 0 - Round 3 -> Round 4")
print("=" * 60)

# Wall ANTES del end_round
wall_p0_before = np.array([
    [ 0, -1, -1, -1, -1],
    [-1, -1, -1, -1, -1],
    [-1, -1, -1, -1,  2],
    [-1, -1,  4, -1, -1],
    [-1, -1, -1, -1, -1]
], dtype=int)

print("\nWall ANTES:")
print(wall_p0_before)

# Pattern lines completadas
print("\nPattern Lines completadas:")
print("  Row 0: [2]      (color 2)")
print("  Row 1: [1 1]    (color 1)")

# Floor line
floor_p0 = np.array([1, 5, 1, 1, 1, 2, 2])
print("\nFloor Line:", floor_p0)

# Simular colocación en muro
wall_p0_after = wall_p0_before.copy()

# Row 0, Color 2
pts_row0, col_row0 = transfer_to_wall_scoring(wall_p0_before, 0, 2)
wall_p0_after[0][col_row0] = 2
print(f"\n1. Colocar color 2 en row 0:")
print(f"   Columna: {col_row0}")
print(f"   Puntos: {pts_row0}")
print(f"   Wall después:")
print(f"   {wall_p0_after[0]}")

# Row 1, Color 1
pts_row1, col_row1 = transfer_to_wall_scoring(wall_p0_after, 1, 1)
wall_p0_after[1][col_row1] = 1
print(f"\n2. Colocar color 1 en row 1:")
print(f"   Columna: {col_row1}")
print(f"   Puntos: {pts_row1}")
print(f"   Wall después:")
print(f"   {wall_p0_after[1]}")

# Penalizaciones
floor_penalty_p0 = calculate_floor_penalization(floor_p0)
print(f"\n3. Penalizaciones Floor Line: {floor_penalty_p0}")

# Total
pts_wall_p0 = pts_row0 + pts_row1
pts_total_p0 = pts_wall_p0 + floor_penalty_p0
score_before_p0 = -9
score_after_p0 = score_before_p0 + pts_total_p0

print(f"\n{'='*40}")
print(f"RESUMEN PLAYER 0:")
print(f"{'='*40}")
print(f"Score antes:        {score_before_p0}")
print(f"Puntos muro:        +{pts_wall_p0} (row0:{pts_row0} + row1:{pts_row1})")
print(f"Penalizaciones:     {floor_penalty_p0}")
print(f"CAMBIO NETO:        {pts_total_p0}")
print(f"Score esperado:     {score_after_p0}")
print(f"Score real (log):   -19")
print(f"DIFERENCIA:         {-19 - score_after_p0}")

print("\n" + "=" * 60)
