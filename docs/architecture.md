
# Arquitectura del modelo AzulNet (Fase 3)

Este documento describe la arquitectura de la red neuronal utilizada en el proyecto Azul Zero, actualizada para la Fase 3 (Diciembre 2025).

## Esquema general

La arquitectura sigue el patrón de AlphaZero pero con mejoras significativas para capturar mejor el contexto del juego y las reglas, dividida en tres bloques de entrada y dos cabezas de salida con un tronco compartido:

- **Entrada Triple**:
    - `x_spatial`: Tablero y estado estructurado (CNN).
    - `x_factories`: Estado de las fábricas (Transformer).
    - `x_global`: Estado global y conteos (MLP).
- **Bloques residuales convolucionales** para la parte espacial.
- **Transformer Encoder** para procesar las fábricas.
- **Positional Encoding**: Información de posición para distinguir fábricas del centro.
- **Tronco Compartido (Shared Trunk)**: Capa de fusión con **LayerNorm** que combina todas las características.
- **Action Masking**: Enmascaramiento aditivo (`logits - 1e9`) directamente en la salida de la política.

![Arquitectura AzulNet](./azul_net_architecture.png)

## Detalle por capas

### Entradas

1.  **`x_spatial`**: Tensor `(batch, in_channels, 5, 5)`. Contiene tableros, muros y líneas de patrón.
2.  **`x_factories`**: Tensor `(batch, N_factories + 1, 5)`. Contiene el conteo de colores en cada fábrica y el centro.
3.  **`x_global`**: Vector `(batch, global_size=34)`. Contiene:
    - Bolsa, descartes, token inicial (11 features).
    - Líneas de suelo y puntuaciones (16 features).
    - **[NUEVO]** Ronda actual normalizada (1 feature).
    - **[NUEVO]** Bonificaciones potenciales: filas, columnas y colores completos por jugador (6 features).
    - **[NUEVO]** Fichas restantes por color (5 features): Suma de bolsa, descartes, fábricas y centro.

### Procesamiento de Fábricas

- **Embedding**: Proyección lineal de 5 colores a `embed_dim` (32).
- **Positional Encoding**: Se suma un vector aprendible `(1, N+1, 32)` para que la red distinga qué entrada es el Centro.
- **Transformer Encoder**: 2 capas de atención para capturar relaciones entre fábricas.
- **Salida**: Vector aplanado de dimensión `(N+1) * embed_dim`.

### Tronco Espacial (Backbone)

- **Conv Inicial**: 64 canales, kernel 3x3.
- **Bloques Residuales**: 4 bloques estándar (Conv3x3 -> BN -> ReLU -> Conv3x3 -> BN -> Add -> ReLU).

### Fusión y Tronco Compartido (Shared Trunk) [NUEVO]

- **Concatenación**: Salida espacial aplanada + Salida de fábricas + `x_global`.
- **Normalización**: **LayerNorm** aplicada al vector concatenado para estabilizar escalas.
- **Fusion Layer (MLP)**:
    - Linear (→ 256) + ReLU
    - Linear (→ 256) + ReLU
- Este bloque permite que la red aprenda interacciones complejas entre las diferentes modalidades de entrada antes de decidir política o valor.

### Rama de Política (Policy Head)

- **Entrada**: Salida del Shared Trunk.
- **MLP**:
    - Linear (Input → 256) + ReLU
    - Linear (→ `action_size`)
- **Action Masking**: Se suma `(mask - 1) * 1e9` a los logits de salida.
    - Acciones legales (1) -> se suma 0.
    - Acciones ilegales (0) -> se suma -1e9 (probabilidad ~0 tras softmax).
- **Salida**: Logits para cada acción posible.

### Rama de Valor (Value Head)

- **Entrada**: Salida del Shared Trunk.
- **MLP**:
    - Linear (→ 256) + ReLU
    - Linear (→ 1)
- **Salida**: **Tanh Activation**. Rango [-1, 1].
    - `+1`: Victoria segura.
    - `-1`: Derrota segura.
    - `0`: Empate.
    - Se cambió de regresión lineal (score difference) a clasificación Win/Loss para mayor estabilidad.

## Diagrama (Mermaid)

```mermaid
graph TD
    subgraph Inputs
        InputSpatial[Spatial (B, 4, 5, 5)]
        InputFactories[Factories (B, 6, 5)]
        InputGlobal[Global (B, 34)]
        ActionMask[Action Mask (B, 180)]
    end

    subgraph Spatial Processing
        InputSpatial --> ConvIn[Conv2D 3x3 (64)]
        ConvIn --> ResBlocks[4x ResBlocks]
        ResBlocks --> PolicyConv[Policy Conv 1x1 (2)]
        ResBlocks --> ValueConv[Value Conv 1x1 (1)]
        PolicyConv --> FlatSpatialP[Flatten]
        ValueConv --> FlatSpatialV[Flatten]
    end

    subgraph Factory Processing
        InputFactories --> Embed[Linear Embed (32)]
        Embed --> AddPos[Add Positional Encoding]
        AddPos --> Transformer[Transformer Encoder (2 layers)]
        Transformer --> FlatFactories[Flatten]
    end

    subgraph Shared Trunk
        FlatSpatialP & FlatFactories & InputGlobal --> Concat[Concat]
        FlatSpatialV --> Concat
        Concat --> LayerNorm[LayerNorm]
        LayerNorm --> FusionFC1[Linear (256) + ReLU]
        FusionFC1 --> FusionFC2[Linear (256) + ReLU]
    end

    subgraph Heads
        FusionFC2 --> PolicyFC1[Linear (256) + ReLU]
        PolicyFC1 --> PolicyLogits[Linear -> Logits]
        ActionMask --> AddMask[Add Mask (-1e9)]
        PolicyLogits --> AddMask
        AddMask --> FinalLogits[Final Logits]

        FusionFC2 --> ValueFC1[Linear (256) + ReLU]
        ValueFC1 --> ValueOut[Linear -> Tanh -> Value]
    end
```

## MCTS & Estrategia de Búsqueda (Fase 3.5)

### Mejoras de Exploración
Para evitar el sobreajuste a estrategias deterministas, se implementaron mecanismos robustos de exploración:

1.  **Dirichlet Noise**:
    - Se añade ruido a los priors del nodo raíz: $P(s,a) = (1-\epsilon)P_{net}(s,a) + \epsilon \text{Dir}(\alpha)$.
    - Parámetros: $\alpha=0.3$, $\epsilon=0.25$.
    - Esto fuerza al MCTS a considerar acciones que la red podría haber descartado prematuramente.

2.  **Temperature Sampling (Dinámico)**:
    - **Rondas 1-2**: $T=1.0$. Exploración alta.
    - **Rondas 3-4**: $T=0.5$. Exploración reducida.
    - **Rondas 5+**: $T=0.0$. Selección voraz (Greedy).
    - Se eliminó el threshold fijo de 30 movimientos en favor de esta lógica basada en fases del juego.

### Optimización: Tree Reuse
- Se implementó la reutilización del árbol de búsqueda entre movimientos.
- Al avanzar el juego, el subárbol correspondiente a la acción tomada se convierte en la nueva raíz.
- **Beneficio**: Preserva estadísticas de visitas y valores, permitiendo una búsqueda más profunda efectiva (~50% más de simulaciones efectivas).

---

## Training Loop & Monitoreo

### Logging Mejorado
El sistema de monitoreo ahora captura métricas críticas para el diagnóstico:

- **Loss Breakdown**: Separación de `Policy Loss` (KL Divergence) y `Value Loss` (MSE) para detectar desbalances.
- **Tensor Shapes**: Corrección de dimensiones en el cálculo de Value Loss para evitar broadcasting incorrecto.
- **MCTS Stats**:
    - `avg_visits`: Verifica la acumulación de visitas.
    - `avg_entropy`: Mide la diversidad de la política (evita colapso determinista).
    - `reuse_rate`: Confirma la efectividad de la reutilización del árbol (>95%).

### Currículum de Entrenamiento
El entrenamiento ajusta dinámicamente los parámetros según el ciclo:

| Ciclo | Simulaciones | Temp Threshold | Noise | Objetivo |
|-------|--------------|----------------|-------|----------|
| 1-6   | 30           | 30             | Sí    | Aprendizaje rápido de reglas |
| 7-16  | 75           | 30             | Sí    | Estrategia y táctica |
| 17+   | 150          | 30             | Sí    | Refinamiento y profundidad |

### Estabilidad del Entrenamiento
Para garantizar una convergencia estable en un espacio de acciones complejo:

1.  **Learning Rate Scheduler**: `CosineAnnealingLR`. El LR decae suavemente dentro de cada ciclo (e.g., $10^{-3} \to 10^{-6}$) para evitar oscilaciones.
2.  **Gradient Clipping**: Se limita la norma de los gradientes a 1.0 (`max_norm=1.0`) para prevenir explosiones de gradiente.

---

## Cambios respecto a Fase 2

1.  **Shared Trunk**: Se añadió un MLP compartido para fusionar características antes de las cabezas.
2.  **Action Mask Injection**: Se inyecta la máscara de acciones legales en la Policy Head.
3.  **Global Input Expandido**: Se añadieron características de ronda y bonificaciones.
4.  **Win/Loss Objective**: Cambio de regresión de puntos a clasificación (-1, 1).
5.  **Robust Exploration**: Implementación de Dirichlet Noise y Temperature Sampling.
6.  **MCTS Tree Reuse**: Persistencia del árbol entre turnos.
7.  **Training Stability**: LR Scheduler y Gradient Clipping.

## Cambios Fase 4 (Arquitectura Robusta)

1.  **Action Masking Additivo**: Cambio de concatenación a enmascaramiento en logits.
2.  **Feature Normalization**: LayerNorm en el Shared Trunk.
3.  **Positional Encoding**: Distinción explícita del Centro en las fábricas.
4.  **Remaining Tiles**: Input explícito de fichas restantes para cálculo de probabilidades.
5.  **Dynamic Temperature**: Exploración adaptativa por rondas.

## Correcciones Críticas (Fase 3.5 - Diciembre 2025)

Se identificaron y corrigieron bugs críticos que afectaban la validez del entrenamiento anterior al 4 de Diciembre de 2025:

1.  **Scoring Bug**:
    *   **Problema**: Se contaba doble la ficha central cuando se completaban líneas horizontales y verticales simultáneamente (`score += v_count + 1`).
    *   **Corrección**: `score += v_count`.
    *   **Impacto**: Scores inflados y recompensas incorrectas.

2.  **State Reset Bug**:
    *   **Problema**: Las `pattern_lines` y `floor_line` no se limpiaban correctamente entre rondas debido a asignación de nuevos arrays en lugar de modificación in-place.
    *   **Corrección**: Uso de slicing `[:] = -1` para mantener referencias de memoria.
    *   **Impacto**: Persistencia de estado inválido entre rondas.

3.  **Training Loop Fixes**:
    *   Corrección de `UserWarning` por mismatch de dimensiones en Value Loss.
    *   Cambio de `CrossEntropy` a `KLDiv` para Policy Loss (mejor manejo de distribuciones).

> [!IMPORTANT]
> Los modelos entrenados antes del commit de corrección (4 Dic 2025) deben considerarse inválidos y descartarse.

---
Última actualización: Diciembre de 2025 (Fase 4.2 - Full Information & Dynamic Temp)