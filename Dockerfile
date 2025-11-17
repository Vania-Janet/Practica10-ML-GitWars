# Imagen base de Python
FROM python:3.11-slim

# Establecer directorio de trabajo
WORKDIR /app

# Copiar archivos de requisitos primero (para cache de Docker)
COPY api/requirements.txt /app/api/requirements.txt

# Instalar dependencias
RUN pip install --no-cache-dir -r /app/api/requirements.txt

# Copiar todo el proyecto al contenedor
COPY . /app

# Crear directorio para el modelo si no existe
RUN mkdir -p /app/api/model

# Exponer puerto 8000
EXPOSE 8000

# Variables de entorno
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app
ENV PORT=8000

# Healthcheck para Docker
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
  CMD python -c "import requests; requests.get('http://localhost:${PORT}/health')" || exit 1

# Comando para ejecutar la aplicación
# Si PORT está definido (Render), usa ese puerto, sino usa 8000
CMD uvicorn api.main:app --host 0.0.0.0 --port ${PORT:-8000}
