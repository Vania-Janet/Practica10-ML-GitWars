# Imagen base de Python
FROM python:3.11-slim

# Establecer directorio de trabajo
WORKDIR /app

# Copiar archivos de requisitos primero (para cache de Docker)
COPY api/requirements.txt ./api/requirements.txt

# Instalar dependencias
RUN pip install --no-cache-dir -r ./api/requirements.txt

# Copiar todo el proyecto al contenedor
COPY . .

# Crear directorio para el modelo si no existe
RUN mkdir -p ./api/model

# Exponer puerto 8000
EXPOSE 8000

# Variables de entorno
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app
ENV PORT=8000

# Comando para ejecutar la aplicación
# Si PORT está definido (Render), usa ese puerto
CMD uvicorn api.main:app --host 0.0.0.0 --port ${PORT}
