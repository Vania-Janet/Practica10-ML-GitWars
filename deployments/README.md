# Configuración de Despliegue

## Descripción

Este directorio contiene los archivos necesarios para el despliegue del proyecto en entornos de producción.

## Archivos

### Dockerfile

Imagen Docker para containerización del servicio.

**Construcción:**
```bash
docker build -t ml-gitwars -f deployments/Dockerfile .
```

**Ejecución:**
```bash
docker run -p 8000:8000 ml-gitwars
```

**Características:**
- Imagen base Python optimizada
- Instalación de dependencias
- Exposición de puerto 8000
- Comando de inicio del servicio

### render.yaml

Configuración para despliegue en Render.

**Componentes configurados:**
- Tipo de servicio (Web Service)
- Comandos de build
- Comandos de inicio
- Variables de entorno
- Health checks

## Despliegue en Render

1. Conectar repositorio de GitHub
2. Seleccionar el archivo `render.yaml`
3. Render detectará automáticamente la configuración
4. El servicio se desplegará automáticamente


## Variables de Entorno

Configurar las siguientes variables según el entorno:
- `MODEL_PATH`: Ruta a modelos entrenados
- `PORT`: Puerto del servicio
- `ENV`: Entorno (development/production)

## Consideraciones

- Mantener imágenes Docker ligeras
- Usar .dockerignore para excluir archivos innecesarios
- Configurar health checks apropiados
- Implementar logging adecuado
- Considerar escalado horizontal según demanda
