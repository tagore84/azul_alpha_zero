#!/usr/bin/env python3
"""
TestEnvAndGameRules.py

Script para verificar exhaustivamente que el entorno (AzulEnv) funciona correctamente.
Ejecuta una partida entre dos RandomPlusPlayer y loguea todos los detalles relevantes
para detectar posibles bugs en:
- Gestión de fichas (factories, center, bag, discard)
- Colocación de fichas en pattern lines y walls
- Cálculo de puntuaciones (durante rondas y al final)
- Penalizaciones de floor line
- Bonificaciones finales
- Clonado de estados
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
from azul.env import AzulEnv
from players.random_plus_player import RandomPlusPlayer

def log_separator(title=""):
    print("\n" + "="*80)
    if title:
        print(f" {title}")
        print("="*80)

def log_state(env, turn, player_idx, action=None):
    """Loguea el estado completo del entorno."""
    print(f"\n--- TURNO {turn} | Jugador {player_idx} ---")
    
    if action:
        print(f"Acción: Source={action[0]}, Color={action[1]}, Dest={action[2]}")
    
    # Bag y Discard
    print(f"Bag: {env.bag} (total: {env.bag.sum()})")
    print(f"Discard: {env.discard} (total: {env.discard.sum()})")
    
    # Factories
    print("\nFactories:")
    for i, factory in enumerate(env.factories):
        if factory.sum() > 0:
            print(f"  F{i}: {factory}")
    
    # Center
    print(f"Center: {env.center} (total: {env.center.sum()})")
    print(f"First Player Token: {env.first_player_token}")
    
    # Players
    for i, p in enumerate(env.players):
        print(f"\nJugador {i}:")
        print(f"  Score: {p['score']}")
        print(f"  Pattern Lines:")
        for row_idx, line in enumerate(p['pattern_lines']):
            print(f"    Row {row_idx} (cap {row_idx+1}): {line}")
        print(f"  Wall:")
        for row in p['wall']:
            print(f"    {row}")
        print(f"  Floor Line: {p['floor_line']}")

def verify_tile_conservation(env, initial_tiles):
    """Verifica que el número total de fichas se conserve."""
    current_tiles = (
        env.bag.sum() +
        env.discard.sum() +
        env.factories.sum() +
        env.center.sum()
    )
    
    # Añadir fichas en pattern lines y floor lines de jugadores
    for p in env.players:
        for line in p['pattern_lines']:
            current_tiles += np.sum(line >= 0)
        current_tiles += np.sum(p['floor_line'] >= 0)
        # Fichas en el muro
        current_tiles += np.sum(p['wall'] >= 0)
    
    if current_tiles != initial_tiles:
        print(f"\n⚠️  WARNING: Tile conservation violated!")
        print(f"  Initial tiles: {initial_tiles}")
        print(f"  Current tiles: {current_tiles}")
        print(f"  Difference: {current_tiles - initial_tiles}")
        return False
    return True

def verify_scores(env):
    """Verifica que las puntuaciones sean razonables."""
    for i, p in enumerate(env.players):
        score = p['score']
        print(f"\nJugador {i} - Score: {score}")
        
        # Las puntuaciones no deberían ser extremadamente altas o bajas sin razón
        if score < -100:
            print(f"  ⚠️  Puntuación muy baja: {score}")
        if score > 200:
            print(f"  ⚠️  Puntuación muy alta: {score}")

def test_game():
    """Ejecuta una partida completa y verifica todo."""
    log_separator("INICIO DE TEST - ENV AND GAME RULES")
    
    # Inicializar entorno y jugadores
    env = AzulEnv()
    obs = env.reset()
    
    player1 = RandomPlusPlayer(name="RandomPlus_P0")
    player2 = RandomPlusPlayer(name="RandomPlus_P1")
    players = [player1, player2]
    
    # Contar fichas iniciales
    initial_tiles = env.bag.sum() + env.discard.sum() + env.factories.sum() + env.center.sum()
    print(f"\nFichas totales iniciales: {initial_tiles}")
    print(f"  Bag: {env.bag.sum()}")
    print(f"  Factories: {env.factories.sum()}")
    
    log_state(env, turn=0, player_idx=obs['current_player'])
    
    turn = 1
    done = False
    round_count = 0
    
    while not done:
        current_player_idx = obs['current_player']
        current_player = players[current_player_idx]
        
        # Obtener acción
        action = current_player.predict(obs)
        if not isinstance(action, tuple):
            action = env.index_to_action(int(action))
        
        # Verificar que la acción es válida
        valid_actions = env.get_valid_actions()
        if action not in valid_actions:
            print(f"\n❌ ERROR: Acción inválida elegida!")
            print(f"  Acción: {action}")
            print(f"  Valid actions count: {len(valid_actions)}")
            print(f"  First 10 valid: {valid_actions[:10]}")
            break
        
        # Ejecutar acción
        log_separator(f"ACCIÓN TURNO {turn}")
        log_state(env, turn, current_player_idx, action)
        
        obs_before = env._get_obs()
        obs, reward, done, info = env.step(action)
        
        print(f"\nPost-Step:")
        print(f"  Reward: {reward}")
        print(f"  Done: {done}")
        print(f"  Info: {info}")
        
        # Log estado después de la acción
        log_state(env, turn, current_player_idx)
        
        # Verificar conservación de fichas
        if not verify_tile_conservation(env, initial_tiles):
            print("\n❌ CRITICAL ERROR: Tile conservation failed!")
            break
        
        # Verificar si cambió de ronda
        if info['round'] > round_count:
            log_separator(f"FIN DE RONDA {round_count} / INICIO RONDA {info['round']}")
            round_count = info['round']
            verify_scores(env)
        
        turn += 1
        
        # Límite de seguridad
        if turn > 500:
            print("\n⚠️  Límite de turnos alcanzado (500), terminando test.")
            break
    
    # Resultado final
    log_separator("RESULTADO FINAL")
    final_scores = env.get_final_scores()
    print(f"\nPuntuaciones finales: P0={final_scores[0]}, P1={final_scores[1]}")
    
    # Verificar fichas finales
    print("\nEstado final del juego:")
    log_state(env, turn, -1)
    
    # Verificación final de conservación
    if verify_tile_conservation(env, initial_tiles):
        print("\n✅ Conservación de fichas: CORRECTA")
    else:
        print("\n❌ Conservación de fichas: FALLIDA")
    
    # Verificar puntuaciones
    verify_scores(env)
    
    # Verificar que al menos un jugador tiene una fila completa
    game_ended_correctly = False
    for p in env.players:
        for row in p['wall']:
            if all(cell != -1 for cell in row):
                game_ended_correctly = True
                break
    
    if done and game_ended_correctly:
        print("\n✅ Juego terminado correctamente (fila completa detectada)")
    elif done:
        print("\n⚠️  Juego terminado pero no se detectó fila completa")
    else:
        print("\n⚠️  Juego no terminó (límite de turnos)")
    
    log_separator("FIN DE TEST")
    
    return final_scores

if __name__ == "__main__":
    try:
        scores = test_game()
        print(f"\n\n🎯 Test completado. Scores: P0={scores[0]}, P1={scores[1]}")
    except Exception as e:
        print(f"\n\n❌ EXCEPCIÓN DURANTE EL TEST:")
        print(f"   {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
