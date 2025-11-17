# Elemento 4: API de Inferencia

## Descripción

API REST minimalista para realizar inferencias con los modelos entrenados.

## Endpoints

### POST /predict
Realiza predicciones con el modelo entrenado.

**Request Body:**
```json
{
  "features": [valor1, valor2, valor3, ...]
}
```

**Response:**
```json
{
  "prediction": valor_predicho,
  "confidence": probabilidad,
  "model": "nombre_modelo"
}
```

### GET /health
Verifica el estado del servicio.

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true
}
```

### GET /models
Lista los modelos disponibles.

**Response:**
```json
{
  "models": ["svm", "random_forest", "mlp"]
}
```

## Inicialización

### Usando Make (Recomendado)

```bash
# Instalar dependencias
make install

# Iniciar servidor de desarrollo
make run

# Ejecutar en modo producción
make serve

# Ver logs
make logs

# Detener servidor
make stop
```

### Comandos Directos

```bash
# Desarrollo
python api/main.py

# Producción con uvicorn (FastAPI)
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload

# Producción con gunicorn (Flask)
gunicorn -w 4 -b 0.0.0.0:8000 api.main:app
```

## Makefile

El proyecto incluye un `Makefile` con los siguientes targets:

- `make install`: Instala dependencias desde requirements.txt
- `make run`: Inicia servidor en modo desarrollo
- `make serve`: Inicia servidor en modo producción
- `make test`: Ejecuta tests de la API
- `make clean`: Limpia archivos temporales
- `make help`: Muestra ayuda de comandos disponibles

## Configuración

Variables de entorno:
- `MODEL_PATH`: Ruta al modelo serializado
- `PORT`: Puerto del servidor (default: 8000)
- `HOST`: Host del servidor (default: 0.0.0.0)
- `RELOAD`: Auto-reload en desarrollo (default: true)

## Ejemplo de Uso

```python
import requests

# Realizar predicción con datos de superhéroe
data = {
    "features": {
        "intelligence": 60,
        "strength": 80,
        "speed": 55,
        "durability": 70,
        "combat": 65,
        "height_cm": 185,
        "weight_kg": 90
    }
}

response = requests.post(
    "http://localhost:8000/predict",
    json=data
)
print(response.json())
```

```bash
# Usando curl
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "features": {
      "intelligence": 60, "strength": 80, "speed": 55,
      "durability": 70, "combat": 65,
      "height_cm": 185, "weight_kg": 90
    }
  }'
```

## Consideraciones

- Los modelos deben estar pre-entrenados y serializados
- Validar formato de entrada antes de predecir
- Implementar manejo de errores robusto
- Considerar rate limiting en producción
