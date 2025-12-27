# Guía de Análisis del Entrenamiento

Este documento detalla cómo monitorear y analizar el proceso de entrenamiento del agente de Azul, utilizando los logs generados por el sistema.

## 1. Logs Principales

La información clave se encuentra en el directorio `logs_v5/`. Los dos archivos más importantes son:

1.  **`logs_v5/training_monitor.log`**: Registro crudo y detallado de cada ciclo y partida.
2.  **`logs_v5/training_analyzed.log`**: Resumen tabular de alto nivel con métricas agregadas por ciclo.

> [!IMPORTANT]
> **Fuente de la Verdad de Hiperparámetros**:
> Antes de realizar cualquier análisis, **DEBES** leer el archivo `scripts/train_loop_v5.py`.
> Específicamente, inspecciona la función `get_curriculum_params(cycle)`.
>
> **Jamás especules** sobre valores de hiperparámetros (ej. *"Si estamos usando pocas simulaciones..."*).
> **Verifica** el valor exacto que se está usando para el ciclo actual en el código.
>
> Hiperparámetros clave a verificar en el código:
> - `simulations`: Número de simulaciones MCTS.
> - `lr`: Learning Rate del optimizador.
> - `cpuct`: Constante de exploración UCB.
> - `n_games`: Partidas de Self-Play por ciclo.
> - `epochs`: Épocas de entrenamiento por ciclo.

---

## 2. Análisis de `training_monitor.log`

Este archivo es útil para depurar comportamientos a nivel de partida individual y verificar la configuración del entrenamiento.

### Información Contenida
- **Parámetros del Ciclo**: Al inicio de cada ciclo se imprimen los hiperparámetros (`lr`, `simulations`, `cpuct`, etc.).
- **Detalle por Partida (Self-Play y Validación)**:
    - Scores finales (P0 vs P1). P0 suele ser el modelo actual en evaluación.
    - Duración de la partida (Rounds y Moves).
    - Ganador (`WIN`/`LOSS` desde la perspectiva de P0) y condición de fin.
    - **Cruzar con `scripts/train_loop_v5.py`**: Verifica que los parámetros impresos coincidan con lo esperado en la función `get_curriculum_params`.
- **Resúmenes de Bloque**:
    - `vs Random`: Resumen agregado de victorias/derrotas contra oponente aleatorio.
    - `vs PreviousCycle`: Resumen agregado contra la versión anterior del modelo.

### Qué comprobar
1.  **Scores Negativos Persistentes**:
    - Si ves scores consistentemente negativos (ej. `-80`, `-50`) en ambos jugadores durante el Self-Play, indica que el modelo no está aprendiendo a puntuar o está siendo demasiado penalizado por movimientos inválidos (aunque el log marque `normal_end`).
2.  **Duración de las Partidas**:
    - Partidas con **muy pocos rounds (menos de 5)** pueden indicar que un jugador se está "suicidando" llenando su línea de suelo rápidamente para terminar la partida (si el incentivo de acabar rápido supera al de ganar puntos).
    - Partidas con **muchos rounds (más de 6)** sugieren un juego estancado con demasiados negativos o dificultad para cerrar. Un promedio superior a 6 consistentemente es mala señal.
3.  **Balance de Victorias**:
    - En Self-Play, el ganador debería alternarse. Si `P0` o `P1` ganan el 100% de las veces, podría haber un sesgo por ser primer jugador (First Player Advantage) significativo o un bug en la lógica de alternancia.

---

## 3. Análisis de `training_analyzed.log`

Este es el archivo principal para ver la **evolución** del aprendizaje. Presenta una tabla con métricas promediadas por ciclo.

> [!WARNING]
> **Datos de Ciclo Incompleto**:
> Si analizas este archivo mientras el entrenamiento está corriendo, la última fila (o filas) pueden contener promedios parciales que no reflejan la realidad final.
> Para confirmar que un ciclo ha terminado, revisa las últimas líneas de `logs_v5/training.log` y verifica que haya empezado el siguiente ciclo o que aparezca "Training Finished".

### Columnas Clave y su Interpretación

| Columna | Significado | Qué buscar (Idealmente) | Alerta Roja |
| :--- | :--- | :--- | :--- |
| **Cycle** | Número de iteración de entrenamiento. | Secuencial. | Reinicios inesperados. |
| **MaxRounds** | % de partidas que acaban por límite de turnos. | N/A (Desactivado). | N/A. |
| **AvgRounds (Tr)** | Duración media (rondas) en Self-Play. | 5-6 rondas. | < 5 (suicidio) o > 6.5 (estancamiento). |
| **AvgScore (Tr)** | Puntuación media en Self-Play. | **Tendencia ascendente**. Debería pasar de negativos/bajos a positivos. | Valores estancados en negativo. |
| **AvgScore (Riv)** | Puntuación del modelo nuevo vs Rival (Random/Greedy). | **Tendencia ascendente**. | Puntuaciones bajas contra rivales débiles. |
| **Diff (Tr)** | Diferencia de puntos (P0 - P1) en Self-Play. | Cercano a 0. Indica competencia equilibrada. | Valores muy altos (First Player Advantage). |
| **Diff (Riv)** | Diferencia de puntos vs Rival. | **Positivo sustancial**. | Negativo o cercano a cero. |
| **Diff (Prev)** | Diferencia de puntos vs Modelo Anterior. | **Positivo**. Indica mejora ciclo a ciclo. | Negativo. |
| **WR (Riv)** | Win Rate vs Rival (Random). | **100% o cercano**. | < 90% contra Random. |
| **WR (Prev)** | Win Rate vs Modelo del ciclo anterior. | **> 50%**. Prueba de mejora continua. | Consistentemente < 50% (catastrophic forgetting). |

### Pasos para el Análisis

0.  **Contextualización (CRÍTICO)**:
    - Abre `scripts/train_loop_v5.py`.
    - Identifica en qué ciclo del "Curriculum" estamos (Warmup, Scaling, High Quality) según el número de ciclo actual.
    - Nota los valores exactos de `lr`, `simulations` y `cpuct`.
    - *Ejemplo: "Estamos en ciclo 8 (Warmup), por lo que usamos 200 sims, cpuct=1.0 y lr=1e-3".*

1.  **Verificar Convergencia ("Learning Curve")**:
    - Mira `AvgScore (Tr)`. ¿Sube ciclo a ciclo?
    - *Bien*: Cycle 1 (20 pts) -> Cycle 10 (45 pts).
    - *Mal*: Cycle 1 (-10 pts) -> Cycle 10 (-12 pts).

2.  **Verificar Superioridad Incremental**:
    - Mira `WR (Prev)`. Esta es la prueba de fuego de la mejora continua.
    - Si `WR (Prev)` oscila alrededor del 50%, el modelo está estancado.
    - Si es consistentemente alto (ej. 60-70%), hay un aprendizaje sólido.

3.  **Comparar Entrenamiento vs Validación**:
    - A veces el modelo "memoriza" cómo ganarse a sí mismo (`Diff (Tr)` estable) pero falla contra externos.
    - Compara `AvgScore (Tr)` con `AvgScore (Riv)`. Si destroza en Self-Play pero pierde contra un Random, está sobreajustado (Overfitting) a su propia política.

---

## Resumen del Checklist de Validación

Al revisar el estado del entrenamiento, responde a estas preguntas:

- [ ] **¿Están subiendo los scores promedio en Self-Play?**
- [ ] **¿El Win Rate contra la versión anterior es > 50%?**
- [ ] **¿Las partidas tienen una duración lógica (5-6 rondas)?**
- [ ] **(Validación Ciclo)**: ¿Has confirmado en `training.log` que el ciclo a analizar ha terminado?
- [ ] **¿Los scores no son consistentemente negativos?**

Si la respuesta a alguna es **NO**, se debe detener el entrenamiento y revisar hiperparámetros (learning rate, peso de exploración Cpuct) o la lógica de recompensas.
