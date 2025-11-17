# Elemento 4: API de Predicción de Poderes de Superhéroes

## Descripción

API REST implementada con FastAPI para realizar predicciones del poder de superhéroes utilizando el modelo Random Forest optimizado con Bayesian Optimization.

## Endpoints Implementados

### GET /health
Verifica el estado del servicio.

**Response:**
```json
{
  "status": "ok"
}
```

**Ejemplo:**
```bash
curl http://localhost:8000/health
# {"status":"ok"}
```

### GET /info
Retorna información del modelo y el equipo.

**Response:**
```json
{
  "team_name": "Equipo GitWars",
  "model_type": "Random Forest",
  "hyperparameters": {
    "n_estimators": 100,
    "max_depth": 8
  },
  "preprocessing": "No requiere escalamiento (Random Forest es invariante a escala)",
  "features": ["intelligence", "strength", "speed", "durability", "combat", "height_cm", "weight_kg"]
}
```

**Ejemplo:**
```bash
curl http://localhost:8000/info
```

### POST /predict
Realiza predicciones del poder de un superhéroe.

**Request Body:**
```json
{
  "features": {
    "intelligence": 50,
    "strength": 80,
    "speed": 60,
    "durability": 70,
    "combat": 55,
    "height": "185 cm",
    "weight_kg": 90
  }
}
```

**Response:**
```json
{
  "prediction": 80.6379187346699
}
```

**Nota:** La altura acepta múltiples formatos: `"185 cm"`, `"6'1"`, `185` (número)

## Inicialización

### Usando Make y Docker (Recomendado)

```bash
# Construir imagen Docker
make build

# Ejecutar contenedor
make run

# Verificar estado
make status

# Ver logs
docker logs superhero-api

# Detener contenedor
make stop
```

### Comandos Directos sin Docker

```bash
# Instalar dependencias
pip install -r api/requirements.txt

# Ejecutar servidor de desarrollo
python -m uvicorn api.main:app --reload --port 8000

# Ejecutar en producción
python -m uvicorn api.main:app --host 0.0.0.0 --port 8000
```

## Makefile

El proyecto incluye un `Makefile` con los siguientes targets principales:

- `make build`: Construye la imagen Docker
- `make run`: Ejecuta el contenedor en segundo plano
- `make status`: Muestra el estado del contenedor
- `make stop`: Detiene y elimina el contenedor
- `make clean`: Limpia imágenes y contenedores
- `make verify`: Verifica que todo esté listo para construir
- `make evaluate`: Simula el flujo de evaluación del docente

## Configuración

Variables de entorno:
- `MODEL_PATH`: Ruta al modelo serializado
- `PORT`: Puerto del servidor (default: 8000)
- `HOST`: Host del servidor (default: 0.0.0.0)
- `RELOAD`: Auto-reload en desarrollo (default: true)

## Ejemplos de Uso

### Usando curl (Linux/macOS/Git Bash)

```bash
# Health check
curl http://localhost:8000/health
# Respuesta: {"status":"ok"}

# Información del modelo
curl http://localhost:8000/info

# Predicción
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "features": {
      "intelligence": 50, "strength": 80, "speed": 60,
      "durability": 70, "combat": 55,
      "height": "185 cm", "weight_kg": 90
    }
  }'
# Respuesta: {"prediction":80.6379187346699}
```

### Usando PowerShell (Windows)

```powershell
# Health check
Invoke-WebRequest -Uri http://localhost:8000/health | Select-Object -ExpandProperty Content

# Información del modelo
Invoke-WebRequest -Uri http://localhost:8000/info | Select-Object -ExpandProperty Content

# Predicción
$body = @{
    features = @{
        intelligence = 50
        strength = 80
        speed = 60
        durability = 70
        combat = 55
        height = "185 cm"
        weight_kg = 90
    }
} | ConvertTo-Json

Invoke-WebRequest -Uri http://localhost:8000/predict -Method POST -Body $body -ContentType "application/json" | Select-Object -ExpandProperty Content
# Respuesta: {"prediction":80.6379187346699}
```

### Usando Python

```python
import requests

# Predicción
data = {
    "features": {
        "intelligence": 50,
        "strength": 80,
        "speed": 60,
        "durability": 70,
        "combat": 55,
        "height": "185 cm",
        "weight_kg": 90
    }
}

response = requests.post("http://localhost:8000/predict", json=data)
print(response.json())
# {'prediction': 80.6379187346699}
```

## Documentación Interactiva

FastAPI proporciona documentación automática e interactiva:

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

Estas interfaces permiten:
- Ver todos los endpoints disponibles
- Probar las APIs directamente desde el navegador
- Ver esquemas de request/response
- Descargar especificación OpenAPI

## Características de la API

- ✅ Validación automática de entrada con Pydantic
- ✅ Preprocesamiento flexible (acepta altura en varios formatos)
- ✅ Healthcheck para Docker y Render
- ✅ Manejo robusto de errores
- ✅ Logging configurado
- ✅ Documentación automática
- ✅ Modelo Random Forest optimizado con Bayesian Optimization

## Estructura de Archivos

```
api/
├── main.py              # Aplicación FastAPI principal
├── requirements.txt     # Dependencias Python
├── verify_build.py      # Script de verificación pre-build
├── model/
│   └── model.pkl        # Modelo Random Forest serializado (1.33 MB)
└── README.md           # Esta documentación
```

## Consideraciones

- El modelo fue entrenado con **600 registros** de superhéroes
- **No requiere escalamiento** (Random Forest es invariante a escala)
- Hiperparámetros optimizados con **Bayesian Optimization**:
  - `n_estimators`: 100 árboles
  - `max_depth`: 8 niveles
- RMSE en test: **16.66**
- La API acepta altura en múltiples formatos para mayor flexibilidad
