# Usa una base oficial compatible con x86_64
FROM --platform=linux/amd64 python:3.11-slim

# Variables para reproducibilidad
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Instala dependencias
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copia el resto del proyecto
COPY . .

# Comando por defecto
CMD ["python", "scripts/train_azul.py"]