# Evaluación del Informe del Experto sobre `step()`

He analizado minuciosamente el código de `src/azul/env.py` contrastándolo con los puntos levantados por el experto. A continuación presento el detalle:

## 1. BUG CRÍTICO: Modificación de Score en max_rounds
**Veredicto: CONFIRMADO ✅**
- **Análisis**: En la línea 183 (`p['score'] -= 25`), el código altera directamente la puntuación del jugador en el estado del juego.
- **Impacto**: Esto corrompe el estado "real" del juego. Un jugador podría perder una partida que legítimamente ganó según las reglas de Azul, solo porque el sistema de entrenamiento quería penalizar la duración.
- **Solución**: La penalización debe aplicarse exclusivamente a la variable `reward` devuelta por `step()`, nunca a `p['score']`.

## 2. BUG: Recompensa Final (Solo jugador activo)
**Veredicto: PARCIALMENTE CORRECTO (Diseño Subóptimo) ⚠️**
- **Análisis**: El código actual `(p['score'] - ...) - 0.5 * (opponent_score - ...)` sí incluye al oponente, por lo que la afirmación "solo considera la puntuación del jugador activo" es técnicamente imprecisa.
- **Problema Real**: El factor `0.5` subestima el impacto de la puntuación del rival, lo cual es inadecuado para un juego de suma cero (o competitivo). Asimismo, usar `score` directamente (que incluye bonus finales) es correcto, pero la fórmula mezcla "delta de score" con "bonus final".
- **Solución**: Adoptar la diferencia directa: `reward = Score_Propio - Score_Rival` (o deltas) para alinear el incentivo con ganar la partida.

## 3. BUG DE LÓGICA: Rotación de Turnos Incorrecta
**Veredicto: FALSO ❌**
- **Análisis**: El experto afirma que el turno cambia incorrectamente. Sin embargo, el código maneja esto explícitamente:
    ```python
    if self._is_round_over():
        done = self._end_round() # Aquí dentro se asigna correctamente self.current_player usando el token de primer jugador
    else:
        self.current_player = opponent # Solo se ejecuta si la ronda NO ha terminado
    ```
- **Conclusión**: La lógica de `if/else` previene que se sobrescriba el turno asignado por `_end_round`. Dado que en Azul alguien *siempre* debe tomar el token -1 (el centro empieza con él y debe vaciarse), `_end_round` siempre encuentra un `first_player_next_round` válido.
- **Nota**: Aunque el código funciona, es frágil. Si `_end_round` fallara en asignar el jugador, el bug existiría. Pero tal como está, **no es un error**.

## 4. BUG: Overflow en floor_line
**Veredicto: CONFIRMADO ✅**
- **Análisis**: En las líneas 161-164, si `floor_line` está lleno (`idxs.size == 0`), las fichas `overflow` se ignoran y no se hace nada con ellas.
- **Impacto**: Las fichas desaparecen del universo del juego (no van a `discard` ni a `bag`). Esto rompe la "conservación de masa" del juego y podría llevar a que el juego se quede sin fichas si se juegan muchas rondas con mucho overflow.
- **Solución**: Las fichas excedentes deben sumarse a `self.discard`.

## 5. PROBLEMA DE DISEÑO: Reward Shaping
**Veredicto: CONFIRMADO ✅**
- **Análisis**: El reward actual es muy "greedy" (puntos inmediatos). AlphaZero funciona mejor con recompensas claras de victoria/derrota o recompensas intermedias que guíen hacia objetivos estratégicos (completar filas) en lugar de solo tácticos.
- **Solución**: Implementar las sugerencias de reward shaping (Bonus por filas, penalización por max_rounds solo en reward, diferencia de puntos).

---

## Resumen de Acción
Recomiendo aplicar correcciones para los puntos **1, 2, 4 y 5**. El punto 3 no requiere cambios de código funcional, aunque una refactorización para claridad no vendría mal.
