# 🧠 Plan de Acción para Mejorar el Entrenamiento de AzulZero  
### _(Ordenado por Prioridad y Optimizado para Aplicación Directa)_

---

## 1️⃣ Reducir el número máximo de rondas (CRÍTICO)
- Ajustar **max_rounds = 6 o 7**.  
- Evita que el modelo aprenda estrategias degeneradas basadas en “aguantar”.  
- Fuerza que el agente **complete filas**, **cierre partidas** y **puntúe**, como en Azul real.

---

## 2️⃣ Ajustar `temp_threshold` para permitir exploración real
- Establecer **temp_threshold = 8**.  
- Permite que los primeros movimientos tengan exploración por temperatura.  
- Evita la política determinista desde la ronda 1, que produce datasets degenerados.

---

## 3️⃣ Incrementar `cpuct` para mejorar el balance entre exploración y explotación
- Ajustar **cpuct = 2.0**.  
- Con 200 simulaciones, este valor permite que MCTS explore más líneas útiles.  
- Corrige el comportamiento excesivamente conservador observado.

---

## 4️⃣ Rebalancear la recompensa para enfatizar cierre y progreso
- Aumentar el peso de completar:
  - Filas  
  - Columnas  
  - Sets de colores  
- Añadir una penalización progresiva si no se completan filas tras varias rondas.  
- Esto enseña al modelo a **cerrar patrones**, no solo a evitar penalizaciones.

---

## 5️⃣ Mantener o aumentar la penalización por alcanzar el límite de rondas
- Refuerza la idea de que terminar la partida pronto es lo óptimo.  
- Evita que el modelo busque “rondas extras” artificiales.

---

## 6️⃣ Introducir ruido en el self-play (solo los primeros movimientos)
- **noise_eps = 0.35**  
- **dirichlet_alpha = 0.3**  
- Solo aplicarlo en los primeros 2–3 movimientos.  
- Esto evita que el self-play colapse en secuencias repetidas.

---

## 7️⃣ Aumentar ligerísimamente el número de simulaciones (opcional)
- Subir a **simulations = 300** si el tiempo lo permite.  
- MCTS podrá ver planes más profundos relacionados con cierre de filas y bonus.

---

## 8️⃣ Verificar que el heurístico no tenga ventajas injustas
- Confirmar que no evalúa bonus finales de forma exacta.  
- Comprobar que no usa reglas del tipo:
  - “si cojo X completo columna -> +7”  
  - “si dejo Y al rival, completa set”  
- Debe jugar bajo las mismas limitaciones que la red.

---

## 9️⃣ Monitorizar si la red comienza a completar filas y provocar finales
- Esto debe empezar a verse entre **los ciclos 9 y 12**.  
- Si no aparece progreso:
  - Reajustar recompensas  
  - Revisar dataset  
  - Revisar política de exploración

---

## 🔟 Mantener dataset grande y evitar acumulación de partidas malas
- Conservar partidas recientes (últimos N ciclos).  
- **Eliminar o reducir** partidas donde:
  - se llega sistemáticamente a max_rounds,  
  - ambos jugadores acaban con puntuaciones negativas,  
  - no se completan filas.  
- Esto elimina *ruido tóxico* del entrenamiento.

---

## ✔ Resumen final del plan
Este conjunto de cambios transforma el modelo desde una política degenerada (“evitar puntos negativos durante 8 rondas”) hacia un estilo de juego genuinamente óptimo de Azul:

- cerrar filas,  
- prevenir columnas del rival,  
- forzar finales,  
- maximizar bonus,  
- jugar agresivo y táctico.

---

¿Quieres que también genere una **versión PDF**, **DOCX**, un **README técnico** o un **diagrama visual del pipeline**?

