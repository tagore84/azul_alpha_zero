
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
- **Tronco Compartido (Shared Trunk)**: Capa de fusión que combina todas las características antes de las cabezas.
- **Action Mask Injection**: Inyección de reglas legales directamente en la cabeza de política.

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

### Procesamiento de Fábricas

- **Embedding**: Proyección lineal de 5 colores a `embed_dim` (32).
- **Transformer Encoder**: 2 capas de atención para capturar relaciones entre fábricas.
- **Salida**: Vector aplanado de dimensión `(N+1) * embed_dim`.

### Tronco Espacial (Backbone)

- **Conv Inicial**: 64 canales, kernel 3x3.
- **Bloques Residuales**: 4 bloques estándar (Conv3x3 -> BN -> ReLU -> Conv3x3 -> BN -> Add -> ReLU).

### Fusión y Tronco Compartido (Shared Trunk) [NUEVO]

- **Concatenación**: Salida espacial aplanada + Salida de fábricas + `x_global`.
- **Fusion Layer (MLP)**:
    - Linear (→ 256) + ReLU
    - Linear (→ 256) + ReLU
- Este bloque permite que la red aprenda interacciones complejas entre las diferentes modalidades de entrada antes de decidir política o valor.

### Rama de Política (Policy Head)

- **Entrada**: Salida del Shared Trunk + **Action Mask** (concatenada).
- **Action Mask**: Vector binario que indica qué acciones son legales. Ayuda a la red a descartar movimientos inválidos.
- **MLP**:
    - Linear (Input + ActionMask → 256) + ReLU
    - Linear (→ `action_size`)
- **Salida**: Logits para cada acción posible.

### Rama de Valor (Value Head)

- **Entrada**: Salida del Shared Trunk.
- **MLP**:
    - Linear (→ 256) + ReLU
    - Linear (→ 1)
- **Salida**: **Valor Lineal** (sin activación Tanh). Representa la diferencia de puntos normalizada.

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
        Embed --> Transformer[Transformer Encoder (2 layers)]
        Transformer --> FlatFactories[Flatten]
    end

    subgraph Shared Trunk
        FlatSpatialP & FlatFactories & InputGlobal --> Concat[Concat]
        FlatSpatialV --> Concat
        Concat --> FusionFC1[Linear (256) + ReLU]
        FusionFC1 --> FusionFC2[Linear (256) + ReLU]
    end

    subgraph Heads
        FusionFC2 & ActionMask --> ConcatPolicy[Concat]
        ConcatPolicy --> PolicyFC1[Linear (256) + ReLU]
        PolicyFC1 --> PolicyOut[Linear -> Logits]

        FusionFC2 --> ValueFC1[Linear (256) + ReLU]
        ValueFC1 --> ValueOut[Linear -> Score Diff]
    end
```

## Cambios respecto a Fase 2

1.  **Shared Trunk**: Se añadió un MLP compartido para fusionar características antes de las cabezas.
2.  **Action Mask Injection**: Se inyecta la máscara de acciones legales en la Policy Head para acelerar el aprendizaje de reglas.
3.  **Global Input Expandido**: Se añadieron características de ronda y bonificaciones (filas/cols completas) para dar contexto estratégico.

---
Última actualización: Diciembre de 2025 (Fase 3)