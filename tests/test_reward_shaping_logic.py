
def simulate_reward_calculation(p0_score, p1_score, p0_floor_moves, p1_floor_moves):
    print(f"--- Escenario: P0={p0_score} pts ({p0_floor_moves} al suelo), P1={p1_score} pts ({p1_floor_moves} al suelo) ---")
    
    # Lógica base (simplificada sin descuentos por rondas para claridad)
    if p0_score > p1_score:
        diff_0 = 1.0
        diff_1 = -1.0
        print("Resultado Base: P0 Gana (+1.0), P1 Pierde (-1.0)")
    elif p1_score > p0_score:
        diff_0 = -1.0
        diff_1 = 1.0
        print("Resultado Base: P0 Pierde (-1.0), P1 Gana (+1.0)")
    else:
        diff_0 = 0.0
        diff_1 = 0.0
        print("Resultado Base: Empate (0.0)")

    # Reward Shaping
    floor_penalty_weight = 0.1
    
    # Aplicación
    penalty_0 = p0_floor_moves * floor_penalty_weight
    penalty_1 = p1_floor_moves * floor_penalty_weight
    
    final_v0 = diff_0 - penalty_0
    final_v1 = diff_1 - penalty_1
    
    print(f"Penalización P0: -{penalty_0:.1f}")
    print(f"Penalización P1: -{penalty_1:.1f}")
    print(f"Valor Final P0 (v para la red): {final_v0:.1f}")
    print(f"Valor Final P1 (v para la red): {final_v1:.1f}")
    print("\n")

if __name__ == "__main__":
    # Caso 1: P0 gana jugando limpio, P1 pierde jugando limpio
    simulate_reward_calculation(50, 40, 0, 0)
    
    # Caso 2: P0 gana pero jugó sucio (3 al suelo), P1 pierde limpio
    simulate_reward_calculation(50, 40, 3, 0)
    
    # Caso 3: P0 pierde y encima jugó sucio (3 al suelo), P1 gana limpio
    simulate_reward_calculation(40, 50, 3, 0)
    
    # Caso 4: Ambos juegan sucio, P0 gana
    simulate_reward_calculation(50, 40, 3, 3)
