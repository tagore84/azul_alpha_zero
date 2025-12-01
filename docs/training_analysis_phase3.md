# Análisis de Entrenamiento - Fase 3 (Ciclos 2-9)

## Resumen General

*   **Ciclos 2-6 (Aprendizaje Básico)**:
    *   **Rival**: Random
    *   **Resultado**: 100% Win Rate.
    *   **Conclusión**: El modelo aprendió rápidamente las reglas básicas y a ganar a un oponente que juega al azar. Esto valida que la red "aprende".

*   **Ciclos 7-9 (Salto Estratégico)**:
    *   **Rival**: RandomPlus (Minimiza penalizaciones de suelo).
    *   **Resultado**: Caída brusca al 0% (Ciclo 7) y leve recuperación al 20% (Ciclos 8-9).
    *   **Observación Clave**: Las puntuaciones del modelo son consistentemente **muy negativas** (entre -20 y -60).

## Análisis Detallado

### El Muro de `RandomPlus`
El cambio de `Random` a `RandomPlus` es un salto de dificultad enorme. `RandomPlus` tiene una heurística simple pero poderosa: *elegir la acción que ponga menos fichas en el suelo*.

*   **Ciclo 7**: El modelo pierde todas las partidas. Sus puntuaciones son desastrosas (-55, -52, -44). Esto indica que el modelo está **comiendo muchas penalizaciones**. Probablemente intenta llenar líneas de patrón sin calcular bien el desbordamiento o toma del centro cuando está muy lleno.
*   **Ciclo 8-9**: Empieza a ganar algunas partidas (1 de 5). Las puntuaciones mejoran ligeramente (menos partidas de -50, más en el rango de -20), pero siguen siendo negativas.

### Diagnóstico: "Ceguera al Suelo"
El modelo actual parece sufrir de "ceguera al suelo". Aunque la red recibe el estado de la línea de suelo, parece que **aún no ha aprendido lo costoso que es llenarla**.

En Azul, las penalizaciones crecen exponencialmente (-1, -2, -4, -6, -8, -11, -14). Llenar el suelo es a menudo peor que no puntuar nada. `RandomPlus` evita esto activamente. El modelo AlphaZero tiene que *descubrir* esto por sí mismo a través de la señal de recompensa (Score Difference).

Al tener puntuaciones tan negativas, la señal de gradiente debería ser fuerte ("¡No hagas eso!"), pero toma tiempo propagarse desde el final del juego hasta las decisiones tempranas que causaron el desastre.

### Evolución del Buffer
*   El buffer ha crecido sanamente hasta **25,000** ejemplos en el Ciclo 9.
*   Esto es bueno: ahora tenemos una base de datos diversa de partidas (incluyendo muchas derrotas contra RandomPlus y partidas internas) para que la red aprenda qué *no* hacer.

## Recomendaciones

1.  **Paciencia (Ciclos 10-15)**: Es normal que el Win Rate se estanque un poco al enfrentar una heurística defensiva fuerte. Con el buffer lleno y la arquitectura mejorada, deberíamos ver una mejora gradual en los próximos 5-10 ciclos.
2.  **Monitorizar Puntuaciones**: Lo más importante ahora no es solo ganar, sino **dejar de tener puntuaciones negativas**. Si en el Ciclo 15 seguimos viendo promedios de -30, habrá que intervenir.
3.  **Posible Ajuste (Futuro)**: Si no mejora, podríamos considerar aumentar el peso de las penalizaciones en la recompensa auxiliar o dar una recompensa intermedia negativa por fichas al suelo (aunque AlphaZero puro prefiere solo recompensa final).

**Estado**: El entrenamiento va por buen camino, aunque está en la fase difícil de "aprender a no perder".
