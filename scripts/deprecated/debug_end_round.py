#!/usr/bin/env python3
"""
Script para debug de _end_round() con logging extensivo
Repite la partida pero con logging detallado del scoring
"""

import sys
import os
import numpy as np

# Add project src folder to PYTHONPATH
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from azul.env import AzulEnv
from azul.rules import transfer_to_wall, calculate_floor_penalization
from players.heuristic_player import HeuristicPlayer

# Monkey patch transfer_to_wall para agregar logging
original_transfer_to_wall = transfer_to_wall

def logged_transfer_to_wall(wall, pattern_line, row):
    """Version con logging de transfer_to_wall"""
    color = pattern_line[0]
    
    # Determinar columna
    row_patterns = [
        [0, 1, 2, 3, 4],  # BLUE, YELLOW, ORANGE, BLACK, RED
        [4, 0, 1, 2, 3],  # RED, BLUE, YELLOW, ORANGE, BLACK
        [3, 4, 0, 1, 2],  # BLACK, RED, BLUE, YELLOW, ORANGE
        [2, 3, 4, 0, 1],  # ORANGE, BLACK, RED, BLUE, YELLOW
        [1, 2, 3, 4, 0],  # YELLOW, ORANGE, BLACK, RED, BLUE
    ]
    col = row_patterns[row].index(color)
    
    print(f"    >>> transfer_to_wall: row={row}, color={color}, col={col}")
    print(f"        Wall ANTES de colocar:")
    for r, wall_row in enumerate(wall):
        print(f"          Row {r}: {wall_row}")
    
    # Llamar al original
    pts = original_transfer_to_wall(wall, pattern_line, row)
    
    print(f"        Wall DESPUÉS de colocar:")
    for r, wall_row in enumerate(wall):
        print(f"          Row {r}: {wall_row}")
    print(f"        Puntos calculados: {pts}")
    
    return pts

# Monkey patch
import azul.rules
azul.rules.transfer_to_wall = logged_transfer_to_wall

# También monkey patch en env
import azul.env
azul.env.transfer_to_wall = logged_transfer_to_wall


def main():
    print("="*70)
    print("DEBUG GAME CON LOGGING EXTENSIVO DE _end_round()")
    print("="*70)
    
    env = AzulEnv(seed=42)  # Usar seed para reproducibilidad
    p1 = HeuristicPlayer()
    p2 = HeuristicPlayer()
    players = [p1, p2]
    
    obs = env.reset()
    done = False
    
    log_file = "debug_end_round_log.txt"
    
    with open(log_file, "w") as f:
        f.write("Debug Game with _end_round() Logging\n")
        f.write("="*70 + "\n\n")
        
        turn_count = 0
        last_round = 1
        
        while not done and turn_count < 200:  # Límite de seguridad
            current_player_idx = env.current_player
            player = players[current_player_idx]
            
            # Detectar cambio de ronda
            if env.round_count != last_round:
                print(f"\n{'='*70}")
                print(f"TRANSICIÓN A ROUND {env.round_count}")
                print(f"{'='*70}\n")
                f.write(f"\n{'='*70}\n")
                f.write(f"TRANSICIÓN A ROUND {env.round_count}\n")
                f.write(f"{'='*70}\n\n")
                last_round = env.round_count
            
            # Get action
            action_raw = player.predict(obs)
            
            if isinstance(action_raw, (int, np.integer)):
                action = env.index_to_action(int(action_raw))
            else:
                action = action_raw
            
            f.write(f"\nTurn {turn_count} | Round {env.round_count} | Player {current_player_idx}\n")
            f.write(f"Action: {action}\n")
            
            # Guardar estado antes del step
            score_before = [p['score'] for p in env.players]
            
            # Detectar si esta acción terminará la ronda
            will_end_round = env._is_round_over()
            if not will_end_round:
                # Verificar si después de esta acción se terminará
                # Simular la acción
                source_idx = action[0]
                if source_idx < env.N:
                    factories_empty = env.factories[source_idx].sum() == env.factories[source_idx, action[1]]
                    other_factories_empty = all(
                        i == source_idx or env.factories[i].sum() == 0 
                        for i in range(env.N)
                    )
                    center_will_be_empty = env.center.sum() == 0
                    will_end_round = factories_empty and other_factories_empty and center_will_be_empty
                else:
                    center_will_be_empty = env.center.sum() == env.center[action[1]]
                    factories_all_empty = all(env.factories[i].sum() == 0 for i in range(env.N))
                    will_end_round = center_will_be_empty and factories_all_empty
            
            if will_end_round:
                print(f"\n{'#'*70}")
                print(f"# ÚLTIMA ACCIÓN DE ROUND {env.round_count} - PLAYER {current_player_idx}")
                print(f"# Action: {action}")
                print(f"{'#'*70}")
                
                # Mostrar estado ANTES de _end_round()
                print(f"\nESTADO ANTES DE step() (que llamará _end_round()):")
                for idx, p in enumerate(env.players):
                    print(f"\n  Player {idx}:")
                    print(f"    Score: {p['score']}")
                    print(f"    Pattern Lines:")
                    for i, line in enumerate(p['pattern_lines']):
                        complete = -1 not in line
                        print(f"      Row {i}: {line} {'← COMPLETA' if complete else ''}")
                    print(f"    Floor Line: {p['floor_line']}")
                
                f.write(f"\n{'#'*70}\n")
                f.write(f"# ÚLTIMA ACCIÓN DE ROUND {env.round_count}\n")
                f.write(f"{'#'*70}\n")
                f.write(f"\nScores ANTES de step(): {score_before}\n")
            
            # Ejecutar step
            obs, reward, done, info = env.step(action)
            
            if will_end_round:
                # Mostrar estado DESPUÉS de _end_round()
                print(f"\nESTADO DESPUÉS DE _end_round():")
                for idx, p in enumerate(env.players):
                    print(f"\n  Player {idx}:")
                    print(f"    Score: {p['score']} (cambio: {p['score'] - score_before[idx]:+d})")
                    print(f"    Pattern Lines:")
                    for i, line in enumerate(p['pattern_lines']):
                        complete = -1 not in line
                        print(f"      Row {i}: {line} {'← COMPLETA' if complete else ''}")
                    print(f"    Floor Line: {p['floor_line']}")
                    print(f"    Wall:")
                    for r, wall_row in enumerate(p['wall']):
                        print(f"      Row {r}: {wall_row}")
                
                score_after = [p['score'] for p in env.players]
                f.write(f"\nScores DESPUÉS de _end_round(): {score_after}\n")
                f.write(f"Cambios: {[after - before for before, after in zip(score_before, score_after)]}\n")
                
                # Esperar confirmación del usuario para continuar
                if env.round_count <= 4:  # Solo para las primeras rondas
                    input(f"\n[Presiona ENTER para continuar a Round {env.round_count}...]")
            
            turn_count += 1
        
        f.write(f"\n\nGame finished after {turn_count} turns\n")
        f.write(f"Final scores: {[p['score'] for p in env.players]}\n")
    
    print(f"\n\nLog guardado en: {log_file}")
    print(f"Final scores: {[p['score'] for p in env.players]}")

if __name__ == "__main__":
    main()
