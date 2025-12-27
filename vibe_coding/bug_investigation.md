# Investigación de Bugs en Entrenamiento Azul Zero

Este documento detalla los posibles puntos de fallo identificados tras el análisis de la regresión en el Ciclo 2 y revisión del código.

## 🔴 Hallazgo Crítico: Ceguera Temporal (Missing Feature)

**Archivo**: `src/azul/env.py`
**Función**: `encode_observation`

El agente **no recibe información sobre la ronda actual**, a pesar de que la documentación de la función dice que sí.

- **Evidencia**: En `src/azul/env.py`, la función `encode_observation` construye el vector `global_parts` con `bag`, `discard`, `scores`, etc., pero **omite explícitamente `round_count`**.
- **Consecuencia**: El agente no tiene "reloj". No puede distinguir entre la Ronda 1 y la Ronda 14.
- **Impacto en el Problema Observable**: El entrenamiento penaliza severamente las partidas que llegan a `MaxRounds` (asignando -1.0 a ambos jugadores). Sin embargo, como el agente es "ciego al tiempo", no puede aprender a acelerar o tomar riesgos calculados cuando se acerca el final. Percibe la penalización de tiempo como ruido aleatorio, lo que lleva a un comportamiento errático o de "zombi" (jugar pasivamente).

## ⚠️ Problema de Estabilidad: Acantilado de Recompensa (Reward Cliff)

**Archivo**: `src/train/self_play.py`
**Lógica**: `max_rounds_reached` Override

Cuando una partida alcanza el límite de rondas, el sistema anula el resultado y asigna `v = -1.0` (derrota total) a **ambos** jugadores.

- **Discontinuidad**:
    - Ronda 14: Jugador A gana por puntos (-150 vs -160). Recompensa: **+0.5** (aprox).
    - Ronda 15 (Límite): Jugador A tiene los mismos puntos. Recompensa: **-1.0** (Override).
- **Conflicto**: Combinado con la "Ceguera Temporal", el agente ve que una estrategia ganadora se convierte repentinamente en una derrota catastrófica sin ninguna señal de aviso en el estado. Esto dificulta enormemente la convergencia.

## ℹ️ Observaciones Menores

1.  **Redundancia en Limpieza de Floor Line**:
    - En `env.py`, método `_end_round`, se hace `p['floor_line'][p['floor_line'] == 5] = -1` justo antes de hacer `p['floor_line'][:] = -1`. La primera línea es redundante. No es un bug funcional, pero ensucia el código.

## Plan de Acción Propuesto

1.  **Arreglar Ceguera Temporal**: Añadir `round_count` (normalizado, ej. `round / max_rounds`) al vector de observación en `env.py`.
2.  **Suavizar Penalización por Tiempo**:
    - En lugar de anular el score con -1.0, aplicar una penalización fuerte pero aditiva al score final (ej. `score - 50`), o;
    - Mantener el override pero asegurar que el agente tenga el input de `round_count` para poder predecirlo. *Recomiendo primero arreglar el input y ver si el agente aprende a evitar el timeout.*

Este documento sirve como base para aplicar correcciones en la siguiente fase.
