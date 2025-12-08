# Plan de Implementación para Recuperación del Entrenamiento (AzulZero)

## Diagnóstico
El análisis del `training.log` y el código actual revela los siguientes problemas críticos:
1. **Estancamiento en Max Rounds**: La mayoría de las partidas terminan por alcanzar `max_rounds` (8) con puntuaciones muy negativas. El modelo está "sobreviviendo" en lugar de jugar.
2. **Exploración Nula**: En `scripts/train_loop_v5.py`, el parámetro `temp_threshold` está establecido a `0`, y en `src/train/self_play.py`, esto fuerza `temperature=0.0` (greedy) para *todos* los movimientos. Esto impide que el modelo explore nuevas estrategias, resultando en un colapso de la política.
3. **Recompensas Insuficientes**: La señal de recompensa actual (diferencia de puntos) es demasiado dispersa y tardía, y con penalizaciones fuertes por `max_rounds`, el modelo quizás prefiere no hacer nada (si eso fuera posible) o entra en bucles.

## Cambios Propuestos

### 1. Configuración del Entrenamiento (`scripts/train_loop_v5.py`)
Ajustar los hiperparámetros del curriculum (Ciclos 6-20) según las recomendaciones del experto y las mejores prácticas.

- **`max_rounds`**: Reducir de 8 a **6**. Esto fuerza partidas más dinámicas y evita el "grindeo" de puntos negativos.
- **`temp_threshold`**: Aumentar a **15 movimientos** (o seguir recomendación de 8, pero 15 asegura apertura variada). *Nota: Se requiere cambio en `self_play.py`*.
- **`cpuct`**: Aumentar de 1.2 a **2.0** para fomentar mayor exploración en el MCTS.
- **`noise_eps`**: Aumentar de 0.25 a **0.35** para mayor ruido en la raíz.

```python
# En get_curriculum_params (Ciclos 6-20)
params['max_rounds'] = 6  # Debe pasarse al main y al Env
params['cpuct'] = 2.0
params['temp_threshold'] = 15 # Exploración primeros 15 movimientos
params['noise_eps'] = 0.35
```

### 2. Lógica de Self-Play (`src/train/self_play.py`)
Corregir la lógica de temperatura para que respete `temp_threshold` como un contador de movimientos, en lugar de la lógica actual basada en rondas que se anulaba con 0.

```python
# Reemplazar lógica de temperatura en play_game
if move_idx < temperature_threshold:
    temp = 1.0
else:
    temp = 0.0
```

### 3. Recompensas y Lógica de Juego (`src/azul/env.py`)
Incentivar explícitamente la finalización de filas para guiar al modelo hacia condiciones de victoria válidas.

- **Bonus por Completar Fila**: En `_end_round` o `step`, añadir una recompensa auxiliar al completar una fila.
- **Implementación**: Detectar cambio en filas completadas en `_end_round` y sumar `+10` (o valor escalado) a la recompensa inmediata.

```python
# En _end_round, calcular filas nuevas completadas y añadir a reward
# reward += new_completed_rows * 10
```

## Plan de Verificación

### Pruebas Manuales
1. **Verificar Configuración**: Ejecutar un script de prueba que instancie `AzulEnv` y `generate_self_play_games` con los nuevos parámetros y verifique (mediante logs) que la temperatura cambia después de N movimientos.
2. **Entrenamiento Corto**: Ejecutar 1 ciclo de entrenamiento (reducido: 10 partidas, 1 época) para asegurar que no hay crash por NaNs o errores de lógica.

### Ejecución
1. Aplicar cambios.
2. Reiniciar el entrenamiento (o continuar desde el último checkpoint válido si es posible, aunque dado el colapso, se recomienda reiniciar el ciclo o curriculum).
