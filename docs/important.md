# Protocolos e Instrucciones Críticas

Este documento sirve como fuente de verdad para protocolos obligatorios y contexto persistente del proyecto.

## 1. Reglas del Juego y Lógica de Negocio
> **Instrucción Mandatoria**: Cualquier tarea relacionada con la lógica, reglas o mecánicas del juego **DEBE** ser validada contra el reglamento oficial.

- **Fuente de Verdad**: [Azul Rulebook](docs/azul_rulebook.md)
- **Acción Requerida**: Antes de modificar o implementar lógica de juego, leer `docs/azul_rulebook.md` para asegurar consistencia.

## 2. Hitos y Versiones (Tags)

*   **v5.0-absolute-verified**: Versión estable con "Absolute Scoring".
    *   **Validación**: El agente aprende a puntuar positivo (AvgScore > 0, WR > Random).
    *   **Estado**: Punto de partida limpio antes de migrar a .Relative Scoring
