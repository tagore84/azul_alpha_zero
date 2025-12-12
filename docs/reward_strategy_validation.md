# Estrategia de Validación: Maximización de Puntos Propios

**Fecha:** 10 de Diciembre de 2025
**Objetivo:** Validar la capacidad de aprendizaje "fundamental" de la red neuronal.

## Contexto del Problema
Analizando los logs de entrenamiento previos (Ciclos 1-7), descubrimos que el agente había aprendido una **"estrategia degenerada"**:
1.  **Stalling (Retraso):** El agente aprendió a extender la partida indefinidamente (aprovechando la falta de un límite de rondas).
2.  **Victoria Relativa:** Al extender la partida, forzaba al rival a cometer errores masivos (e.g., -500 puntos) mientras él cometía errores ligeramente menores (e.g., -480 puntos).
3.  **Resultado:** Win Rate del 100% (técnicamente "ganaba"), pero con puntuaciones ridículamente negativas. Esto indica que no estaba aprendiendo a jugar a Azul (completar filas, sumar puntos), sino a explotar un loophole matemático del entorno.

## La Hipótesis
Creemos que la función de recompensa original (`+1` ganar, `-1` perder) es demasiado **"dispersa" (sparse)** y, combinada con el loophole de las rondas infinitas, llevó al agente a una optimización local errónea.

Antes de reintroducir la competición (Zero-Sum), necesitamos responder a una pregunta fundamental:
> **¿Es capaz esta arquitectura de red + MCTS de aprender simplemente a sumar puntos?**

Si la red no puede aprender a maximizar su propia puntuación en un entorno aislado (o ignorando al rival), entonces tenemos un problema más profundo (arquitectura, representación de entrada, bugs en MCTS).

## La Modificación (Reward Shaping)
Hemos modificado `src/train/self_play.py` para cambiar el **Value Target (`v`)**:

- **Antes:**
    - `v = +1` (Si Score_Propio > Score_Rival)
    - `v = -1` (Si Score_Propio < Score_Rival)
- **Ahora:**
    - `v = clamp(Score_Propio / 100.0, -1.0, 1.0)`

Esto elimina el componente adversario. La red ahora recibe una señal de feedback que escala linealmente con la calidad de su juego. Una puntuación de `20` es mejor que `10`, y `50` es mejor que `20`. Antes, ganar por 1 punto o por 100 daba la misma recompensa.

## Criterios de Éxito
Consideraremos que la validación ha sido exitosa si, tras 2-3 ciclos de entrenamiento, observamos:
1.  **Tendencia Positiva de Puntuaciones:** La puntuación media (vs Random o Heurístico) debe subir consistentemente (de -30 a +20, +50, etc.).
2.  **Reducción de Penalizaciones:** El agente debe dejar de llenar su "floor line" intencionadamente.

## Guía de Análisis de Fallo
**¿Qué pasa si la puntuación NO mejora (o sigue siendo negativa)?**

Si después de varios ciclos con este nuevo reward la red sigue sin sumar puntos positivos, el problema NO es la estrategia de juego, sino algo estructural. Debemos investigar en este orden:

1.  **Representación del Estado (Inputs):** ¿Está la red "ciega"?
    - Verificar `env.encode_observation()`. ¿Están los canales de los tableros y factorías correctamente normalizados y orientados?
2.  **Capacidad de la Red:** ¿Es la red demasiado pequeña o tiene un cuello de botella?
    - Revisar el tamaño de los embeddings de las factorías o el número de bloques residuales.
3.  **Bugs en MCTS:**
    - ¿Se está explorando correctamente el árbol? Verificar `cpuct` y el ruido de Dirichlet.
    - ¿Se están propagando bien los valores `v` por el árbol?
4.  **Overfitting / Olvido:**
    - ¿Está el buffer de repetición dominado por partidas antiguas "malas"? (Quizás necesitemos purgar el buffer).

---
**Siguientes Pasos (si es exitoso):**
Una vez validado que la red maximiza puntos, volveremos a una función de recompensa competitiva (e.g., `Score_Diff` o `Win/Loss` con penalización por duración), sabiendo que la base es sólida.
