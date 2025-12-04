# Plan de Futuro - Azul Zero

Este documento detalla el plan de acción para las próximas iteraciones del proyecto, enfocándose en el escalado y refinamiento tras completar todas las mejoras arquitectónicas críticas (Fases 3.5, 4.0, 4.1 y 4.2).

## 1. Fase 5.0: Escalado de Entrenamiento (Prioridad Alta)

Con una arquitectura robusta y libre de bugs, el objetivo ahora es maximizar la calidad del aprendizaje mediante el escalado de recursos.

### 1.1 Aumento de Simulaciones MCTS
- **Actual:** 30/75/150 simulaciones.
- **Objetivo:** Aumentar a 100/200/400 para mejorar la calidad de los targets ($\pi$).
- **Impacto:** Targets más fuertes reducen el ruido y aceleran la convergencia a un juego óptimo.

### 1.2 Aumento de Volumen de Juegos
- **Actual:** 50-200 juegos por ciclo.
- **Objetivo:** Aumentar a 500-1000 juegos por ciclo.
- **Impacto:** Mayor diversidad de estados en el Replay Buffer, reduciendo el sobreajuste.

## 2. Mejoras Pendientes / A Evaluar (Prioridad Baja)

Mejoras que podrían aportar valor marginal o requieren experimentación cuidadosa.

### 2.1 Value Head Lineal (Sin Tanh)
- **Análisis:** AlphaZero usa Tanh para rango [-1, 1]. Cambiar a lineal podría ayudar con gradientes pero requiere cambiar targets y loss. **Decisión:** Mantener Tanh por ahora.

### 2.2 Early Resignation
- **Análisis:** Útil para ahorrar cómputo en self-play, pero riesgo de sesgar datos si no se calibra bien. **Decisión:** Posponer hasta tener un modelo muy fuerte.

### 2.3 Features Explícitas de Muro
- **Análisis:** Indicar qué color va en cada casilla del muro. La red CNN debería aprender esto trivialmente. **Decisión:** Baja prioridad.

---

## Roadmap Sugerido

1.  **Fase 5.0 (Próxima):** Iniciar entrenamiento a gran escala con la nueva arquitectura.
2.  **Fase 5.1:** Monitoreo y ajuste de hiperparámetros (LR, weight decay).
