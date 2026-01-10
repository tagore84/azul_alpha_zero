# Guía de Análisis de Entrenamiento Azul Zero (V6)

Esta guía establece el protocolo estándar para monitorear y analizar el progreso del entrenamiento de Azul Zero. El objetivo es detectar anomalías a tiempo (como el "Equilibrio de Suicidio") y garantizar que los modelos progresan hacia un juego de alta calidad.

## 1. Rutina de Análisis

Cada vez que se desee verificar el estado del entrenamiento, sigue estos pasos estrictos:

### Paso 1: Generar Datos Actualizados
Ejecuta los scripts de análisis y visualización desde la raíz del proyecto:

```bash
# 1. Analizar logs de texto y generar training_analyzed.log
python3 scripts/analyze_logs.py --dir logs_v6

# 2. Generar gráficas de progreso (training_progress.png)
python3 scripts/visualize_progress.py --dir logs_v6
```

### Paso 2: Crear el Informe
1.  Copia la plantilla `docs/templates/analysis_report_template.md`.
2.  Crea un nuevo archivo en `docs/training_analysis/YYYY-MM-DD-ID_analysis.md` (donde ID es secuencial: 1, 2, 3...).
3.  Rellena el informe siguiendo las secciones e interpretando los datos generados.

---

## 2. Semáforos y Umbrales

Utiliza estos indicadores para evaluar la salud del entrenamiento.

### 🟢 Verde (Saludable)
Todo marcha según lo previsto.
*   **Win Rate vs Rival**: Tendencia ascendente o estable > 55%.
*   **Avg Score (Self-Play)**: Positivo (> 0) y creciente. Indica que los agentes buscan ganar puntos, no solo molestar.
*   **Diff Rival**: Positiva.
*   **Training Loss**: Decreciente o estabilizada en valores bajos (< 2.0).

### 🟡 Amarillo (Atención)
Requiere vigilancia, pero no detener el entrenamiento.
*   **Win Rate Estancado**: Se mantiene en 50% +/- 5% durante > 5 ciclos. (Puede indicar meseta de aprendizaje).
*   **Score Fluctuante**: El score promedio sube y baja erráticamente.
*   **Rival Débil**: El modelo tiene WinRate alto (90%) pero el rival es `RandomPlus` (Fase 1). **No confundir con excelencia**.
*   **Loss Value Head**: Sube repentinamente. (El "crítico" está confundido).

### 🔴 Rojo (Crítico - Acción Inmediata)
Detener análisis y plantear intervención.
*   **Win Rate < 10%**: El modelo ha olvidado cómo jugar o ha entrado en una estrategia perdedora.
*   **Avg Score Negativo Persistente (< -5)**: **ALERTA DE SUICIDIO**. Indica que el modelo prioriza minimizar la diferencia relativa en lugar de maximizar su puntuación absoluta.
*   **Partidas Infinitas (Max Rounds)**: Si el % de Max Rounds supera el 20%, el modelo no sabe cerrar la partida.

---

## 3. Diccionario de Métricas

Explicación detallada de las columnas en `training_analyzed.log`.

| Métrica | Descripción | Interpretación Ideal |
| :--- | :--- | :--- |
| **Cycle** | Número de iteración del bucle de entrenamiento. | - |
| **RivalName** | **CRÍTICO**. Contra quién jugamos. | `RandomPlus` (Fácil) vs `Heuristic` (Medio) vs `MinMaxDepth1` (Duro). |
| **MaxRounds** | % de partidas que agotaron el límite de turnos sin terminar. | < 5%. Si es alto, el modelo juega pasivo. |
| **AvgRounds** | Duración promedio de la partida. | ~5-7 rondas es normal en Azul. |
| **AvgScore (Tr)** | Puntuación media en partidas de Self-Play. | **Debe ser POSITIVA**. Si es negativa, alerta roja. |
| **Diff (Tr)** | Diferencia de puntos (P0 - P1) en Self-Play. | Debe ser cercana a 0 (equilibrio entre clones). |
| **AvgScore (Riv)** | Puntuación propia contra el Rival de validación. | Cuanto más alta, mejor. |
| **Diff (Riv)** | Diferencia (Modelo - Rival). | Positiva indica superioridad. |
| **WR (Riv)** | Win Rate contra el Rival (Heurístico/MinMax). | **CONTEXTO**: 90% vs Random es normal. 90% vs Heuristic es divino. |
| **WR (Prev)** | Win Rate contra el modelo del ciclo anterior. | > 50% indica que estamos aprendiendo (o al menos cambiando). |

---

## 4. Acciones de Recuperación

Si el semáforo está en **ROJO**, considera estas acciones en el informe:

1.  **Ajuste de Recompensa**: Si el Score es negativo, revisar `self_play.py` para aumentar el peso de la recompensa absoluta.
2.  **Resetear Buffer**: Si el modelo está "sobre-entrenado" en malas partidas, borrar `replay_buffer.pt`.
3.  **Aumentar Exploración**: Subir `cpuct` o `noise_epsilon` en `train_loop_v6.py`.
4.  **Soft Reset**: Cargar un checkpoint anterior (verde) y reiniciar con parámetros más conservadores.
