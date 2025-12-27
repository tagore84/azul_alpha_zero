# Reporte de Análisis del Entrenamiento (Ciclo 13)

**Fecha**: 2025-12-26
**Estado**: En progreso (Validación Ciclo 13)

## 1. Resumen de Estado
El entrenamiento se encuentra en la fase de **Warmup Avanzado** (Ciclos 11-15).
- **Ciclo Actual**: 13
- **Fase**: Validación (vs PreviousCycle).

## 2. Verificación de Hiperparámetros
Según `scripts/train_loop_v5.py`, para el Ciclo 13 corresponden:
- **Simulaciones**: 300
- **Juegos Self-Play**: 100
- **Validation Opponent**: Heuristic
- **CPUCT**: 1.75
- **Learning Rate**: 5e-4

## 3. Análisis de Métricas (Logs)

### A. Evolución de Puntuación (Score)
| Ciclo | AvgScore (Self-Play) | AvgScore (Rival) | Observación |
| :--- | :--- | :--- | :--- |
| 10 | -5.5 | 15.5 | |
| 11 | -4.0 | 24.5 | Mejora notable vs Rival |
| 12 | -5.3 | 24.5 | Estancamiento ligero |
| 13 | **-9.5** | **29.2** | **Divergencia interesante** |

**Análisis**:
- Los scores en Self-Play son consistentemente negativos (-5 a -10). Sin embargo, esto **no** parece indicar un fallo catastrófico ("suicidio"), ya que la duración de las partidas (`AvgRounds`) se mantiene saludable en **5.5 - 5.8 rondas**.
- El score contra el Rival (Heurístico) está en su punto más alto (29.2 pts). Esto confirma que el modelo **sí sabe puntuar**, pero en Self-Play se neutralizan mutuamente o juegan de forma muy agresiva penalizándose.

### B. Tasas de Victoria (Win Rates)
- **vs Rival (`WR Riv`)**:
    - Ciclo 12: 80%
    - Ciclo 13: 70% (Mantiene superioridad clara).
- **vs Versión Anterior (`WR Prev`)**:
    - Ciclo 12: 40% (Bajada, posible inestabilidad).
    - Ciclo 13: **66.7%** (Recuperación en curso, 4 victorias de 6 jugadas reportadas en monitoreo).

## 4. Conclusiones y Recomendaciones
1.  **Salud del Entrenamiento**: **BUENA**. A pesar de los scores negativos en self-play, el modelo derrota consistentemente a la heurística y está ganando a su versión anterior en el ciclo 13.
2.  **Sobre los Negativos**: Dado que `AvgRounds` es normal (~5.8), los negativos en self-play probablemente se deban a una defensa agresiva (llenar la pila de descartes/suelo del oponente) más que a incompetencia.
3.  **Acción**:
    - **Continuar el entrenamiento**.
    - Monitorear que `WR (Prev)` en el Ciclo 13 termine por encima del 50% (actualmente ~66%).
    - Si el `AvgScore (Tr)` cae por debajo de -15 o `AvgRounds` baja de 5.0, considerar revisar penalizaciones.

---
*Generado automáticamente por Antigravity a partir de `logs_v5/training_analyzed.log` y `logs_v5/training_monitor.log`.*
