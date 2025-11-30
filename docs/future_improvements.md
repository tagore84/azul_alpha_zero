# Mejoras Futuras para Azul AlphaZero

Este documento recopila propuestas de optimización y mejora para la arquitectura y el entrenamiento de AzulNet, basadas en el análisis crítico de la implementación actual (Noviembre 2025).

Estas mejoras están pensadas para una "Fase 2" una vez que la implementación base (AlphaZero estándar) sea estable.

## 1. Optimización del Value Head (Prioridad Alta)

### Problema Actual
Actualmente, el *Value Head* utiliza una activación `Tanh` que produce un valor en el rango `[-1, 1]`. El entrenamiento utiliza el resultado de la partida (Ganar=1, Perder=-1, Empate=0) como target.
En Azul, ganar por 1 punto es muy diferente a ganar por 50 puntos. La señal binaria pierde mucha información sobre la calidad de la victoria.

### Propuesta
Cambiar el objetivo de predicción a **Diferencia de Puntos Normalizada**.

*   **Arquitectura**: Eliminar `Tanh` final o usar una activación lineal.
*   **Target**: `(Puntuación_Propia - Puntuación_Rival) / Factor_Normalización`.
    *   *Factor_Normalización* podría ser 100 (máxima diferencia razonable).
*   **Beneficio**: El modelo aprenderá a maximizar su ventaja y minimizar la del rival, no solo a "ganar por la mínima", lo que lleva a un juego más robusto.

## 2. Representación del Estado Global (Prioridad Media)

### Problema Actual
La información de las fábricas (`factories`) se aplana (`flatten`) en el vector global. Esto destruye la estructura "local" de cada fábrica (qué fichas están juntas en la misma fábrica).

### Propuesta
Representar las fábricas y el centro como tensores estructurados o usar embeddings.

*   **Opción A (Tensor)**: Añadir un input `x_factories` de dimensión `(Batch, N_Fabricas, N_Colores)` o similar, y procesarlo con capas densas específicas antes de concatenarlo al global.
*   **Opción B (Embeddings)**: Usar embeddings para representar el contenido de cada fábrica.
*   **Beneficio**: La red podrá entender mejor relaciones como "si cojo rojo de la fábrica 1, dejo las azules para el rival", que es difícil de ver en un vector plano.

## 3. Arquitectura con Atención / Transformers (Prioridad Media-Baja)

### Problema Actual
La red usa CNNs (convoluciones) que son excelentes para relaciones espaciales (tablero 5x5), pero mediocres para relacionar entidades disjuntas (fábricas, bolsa, rival). La concatenación del vector global es una solución simple pero limitada.

### Propuesta
Introducir un pequeño módulo de **Self-Attention** (Transformer Encoder) para la parte no espacial.

*   **Implementación**: Tratar cada fábrica, el centro, y el estado propio/rival como "tokens" que interactúan entre sí mediante atención.
*   **Beneficio**: Capturar dependencias complejas de largo alcance y lógica combinatoria entre fábricas y tablero.

## 4. Features Globales Explícitas (Prioridad Baja)

### Problema Actual
El vector global contiene conteos crudos.

### Propuesta
Añadir features derivadas explícitas que ayuden a la red:
*   Probabilidad de aparición de cada color (basado en bolsa + descartes).
*   Diferencia de puntos actual.
*   Número de fichas necesarias para completar columnas/filas específicas.

---
**Nota**: Antes de abordar estas mejoras, es crucial validar que el modelo base (Ciclo 1-3) es capaz de jugar partidas legales y terminar correctamente.
