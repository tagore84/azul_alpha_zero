#!/usr/bin/env python3
"""
Script para verificar las correcciones de bugs aplicadas
"""

import sys
import os
import numpy as np

# Add project src folder to PYTHONPATH
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from azul.env import AzulEnv
from azul.rules import transfer_to_wall
from players.heuristic_min_max_mcts_player import HeuristicPlayer

def test_scoring_fix():
    """Verificar que el scoring ya no suma v_count + 1"""
    print("="*70)
    print("TEST 1: Verificación del Fix de Scoring")
    print("="*70)
    
    # Crear un muro con una ficha para probar scoring vertical
    wall = np.full((5, 5), -1, dtype=int)
    wall[0][2] = 2  # Color 2 en row 0, col 2
    
    # Pattern line completa: [1, 1] para row 1
    # Color 1 va a col 2 (según row_patterns row 1)
    pattern_line = np.array([1, 1], dtype=int)
    
    print("\nWall ANTES de transfer_to_wall:")
    print(wall)
    print(f"\nPattern line: {pattern_line} (row 1, color 1)")
    
    pts = transfer_to_wall(wall, pattern_line, row=1)
    
    print("\nWall DESPUÉS de transfer_to_wall:")
    print(wall)
    print(f"\nPuntos calculados: {pts}")
    
    # Verificar:
    # - Horizontal: solo la ficha = 1 punto
    # - Vertical: row 0 col 2 + row 1 col 2 = 2 fichas
    # - Con el bug: score = 1 + (1 + 1) = 3
    # - SIN bug (correcto): score = 1 + 1 = 2
    
    expected_with_bug = 3
    expected_correct = 2
    
    if pts == expected_correct:
        print(f"\n✅ CORRECTO: Scoring calculó {pts} puntos (esperado: {expected_correct})")
        print("   El fix está funcionando - no suma v_count + 1")
        return True
    elif pts == expected_with_bug:
        print(f"\n❌ ERROR: Scoring calculó {pts} puntos (bug aún presente)")
        print(f"   Esperado: {expected_correct} puntos")
        return False
    else:
        print(f"\n⚠️  INESPERADO: Scoring calculó {pts} puntos")
        print(f"   Esperado: {expected_correct} o {expected_with_bug}")
        return False

def test_array_assignments():
    """Verificar que las asignaciones in-place funcionan"""
    print("\n\n" + "="*70)
    print("TEST 2: Verificación de Asignaciones de Arrays")
    print("="*70)
    
    env = AzulEnv(seed=123)
    env.reset()
    
    # Simular un juego simple hasta fin de ronda
    p1 = HeuristicPlayer()
    p2 = HeuristicPlayer()
    players = [p1, p2]
    
    obs = env.reset()
    done = False
    turn = 0
    
    print("\nSimulando juego hasta fin de Round 1...")
    
    while not done and turn < 50:
        player = players[env.current_player]
        action = player.predict(obs)
        
        if isinstance(action, (int, np.integer)):
            action = env.index_to_action(int(action))
        
        obs, reward, done, info = env.step(action)
        turn += 1
        
        # Verificar después de la primera ronda
        if env.round_count == 2:
            print(f"\n✅ Round 1 completado en {turn} turnos")
            
            # Verificar que pattern lines y floor lines estén limpios
            for idx, p in enumerate(env.players):
                pattern_clean = all(
                    all(slot == -1 for slot in line) 
                    for line in p['pattern_lines']
                )
                floor_clean = all(slot == -1 for slot in p['floor_line'])
                
                print(f"\nPlayer {idx}:")
                print(f"  Pattern Lines limpias: {'✅' if pattern_clean else '❌'}")
                print(f"  Floor Line limpia: {'✅' if floor_clean else '❌'}")
                
                if not pattern_clean:
                    print(f"  Pattern Lines: {p['pattern_lines']}")
                if not floor_clean:
                    print(f"  Floor Line: {p['floor_line']}")
            
            all_clean = all(
                all(all(slot == -1 for slot in line) for line in p['pattern_lines']) and
                all(slot == -1 for slot in p['floor_line'])
                for p in env.players
            )
            
            if all_clean:
                print("\n✅ CORRECTO: Arrays se limpiaron correctamente después de Round 1")
                return True
            else:
                print("\n❌ ERROR: Algunos arrays no se limpiaron")
                return False
    
    print("\n⚠️  Game terminó sin completar Round 1")
    return False

def test_complete_game():
    """Ejecutar un juego completo y verificar consistencia"""
    print("\n\n" + "="*70)
    print("TEST 3: Juego Completo")
    print("="*70)
    
    env = AzulEnv(seed=456)
    p1 = HeuristicPlayer()
    p2 = HeuristicPlayer()
    players = [p1, p2]
    
    obs = env.reset()
    done = False
    turn = 0
    max_turns = 500
    
    print("\nEjecutando juego completo...")
    
    while not done and turn < max_turns:
        player = players[env.current_player]
        action = player.predict(obs)
        
        if isinstance(action, (int, np.integer)):
            action = env.index_to_action(int(action))
        
        obs, reward, done, info = env.step(action)
        turn += 1
    
    if done:
        final_scores = [p['score'] for p in env.players]
        print(f"\n✅ Juego completado en {turn} turnos")
        print(f"   Rondas jugadas: {env.round_count - 1}")
        print(f"   Scores finales: {final_scores}")
        
        # Verificar que los scores sean razonables (no negativos extremos)
        if all(score > -100 for score in final_scores):
            print("   Scores parecen razonables")
            return True
        else:
            print("   ⚠️  Scores muy negativos, posible problema")
            return False
    else:
        print(f"\n⚠️  Juego no terminó después de {max_turns} turnos")
        return False

def main():
    print("\n" + "="*70)
    print("VERIFICACIÓN DE FIXES DE BUGS")
    print("="*70)
    
    results = []
    
    # Test 1: Scoring fix
    try:
        results.append(("Scoring Fix", test_scoring_fix()))
    except Exception as e:
        print(f"\n❌ ERROR en test de scoring: {e}")
        results.append(("Scoring Fix", False))
    
    # Test 2: Array assignments
    try:
        results.append(("Array Assignments", test_array_assignments()))
    except Exception as e:
        print(f"\n❌ ERROR en test de arrays: {e}")
        results.append(("Array Assignments", False))
    
    # Test 3: Complete game
    try:
        results.append(("Complete Game", test_complete_game()))
    except Exception as e:
        print(f"\n❌ ERROR en test de juego completo: {e}")
        results.append(("Complete Game", False))
    
    # Resumen
    print("\n\n" + "="*70)
    print("RESUMEN DE TESTS")
    print("="*70)
    
    for test_name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status} - {test_name}")
    
    all_passed = all(passed for _, passed in results)
    
    print("\n" + "="*70)
    if all_passed:
        print("✅ TODOS LOS TESTS PASARON")
        print("Los fixes están funcionando correctamente")
    else:
        print("❌ ALGUNOS TESTS FALLARON")
        print("Revisar los errores arriba")
    print("="*70)
    
    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(main())
