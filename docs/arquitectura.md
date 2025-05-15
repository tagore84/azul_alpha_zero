

# Arquitectura del modelo AzulNet

Este documento describe la arquitectura de la red neuronal utilizada en el proyecto Azul Zero.

## Esquema general

La arquitectura sigue el patrón de AlphaZero, dividida en tres bloques principales:

- Entrada dual: `x_spatial` (tablero y estado estructurado) y `x_global` (estado global)
- Bloques residuales convolucionales
- Cabezas separadas para política (policy head) y valor (value head)

![Arquitectura AzulNet](./azul_net_architecture.png)

## Detalle por capas

### Entrada

- `x_spatial`: tensor de dimensión `(batch_size, in_channels, 5, 5)`
- `x_global`: vector de dimensión `(batch_size, global_size)`

### Convolución inicial

- Conv2D: canales de entrada → 64, kernel 3x3, padding=1
- BatchNorm2D
- ReLU

### Bloques residuales (4 por defecto)

Cada bloque:
- Conv2D 64→64, kernel 3x3, padding=1
- BatchNorm2D → ReLU
- Conv2D 64→64, kernel 3x3, padding=1
- BatchNorm2D
- Suma residual + ReLU

### Rama de política (policy head)

- Conv2D 64→2, kernel 1x1
- BatchNorm2D
- Aplanado
- Concatenación con `x_global`
- Linear → logits para cada acción posible (`action_size`)

### Rama de valor (value head)

- Conv2D 64→1, kernel 1x1
- BatchNorm2D
- Aplanado
- Concatenación con `x_global`
- Linear (→ 256) → ReLU
- Linear → Tanh (valor final entre -1 y 1)

## Observaciones

- La arquitectura busca un balance entre capacidad y velocidad, adecuada para ejecución en CPU (MacBook M1).
- El diseño modular permite ajustar `num_blocks` o `channels` según necesidad.

---
Última actualización: mayo de 2025