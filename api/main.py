"""
API REST para predicción de poder de superhéroes.
Implementa endpoints para verificación de salud, información del modelo y predicción.
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import pickle
import numpy as np
from pathlib import Path
import re
from typing import Dict, Any
import logging

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Inicializar FastAPI
app = FastAPI(
    title="Superhero Power Prediction API",
    description="API para predecir el poder de superhéroes usando el mejor modelo optimizado",
    version="1.0.0"
)

# Variables globales para el modelo y preprocesador
model = None
scaler = None
model_info = {}


class Features(BaseModel):
    """Modelo de datos para las características de entrada."""
    intelligence: float = Field(..., description="Nivel de inteligencia")
    strength: float = Field(..., description="Nivel de fuerza")
    speed: float = Field(..., description="Nivel de velocidad")
    durability: float = Field(..., description="Nivel de durabilidad")
    combat: float = Field(..., description="Nivel de combate")
    height: Any = Field(..., alias="height", description="Altura (puede ser número o string como '6\\'8' o '203 cm')")
    weight_kg: Any = Field(..., description="Peso (puede ser número o string como '980 lb' o '445 kg')")

    class Config:
        populate_by_name = True
        json_schema_extra = {
            "example": {
                "intelligence": 60,
                "strength": 80,
                "speed": 55,
                "durability": 70,
                "combat": 65,
                "height": "6'8",
                "weight_kg": "980 lb"
            }
        }


class PredictRequest(BaseModel):
    """Modelo de datos para la solicitud de predicción."""
    features: Features


class PredictResponse(BaseModel):
    """Modelo de datos para la respuesta de predicción."""
    prediction: float


def parse_height(height_value: Any) -> float:
    """
    Convierte diferentes formatos de altura a centímetros.
    
    Args:
        height_value: Puede ser float, int, o string (ej: "6'8", "203 cm", "203")
    
    Returns:
        float: Altura en centímetros
    """
    if isinstance(height_value, (int, float)):
        return float(height_value)
    
    if isinstance(height_value, str):
        height_str = height_value.strip()
        
        # Formato feet'inches (ej: "6'8")
        if "'" in height_str:
            match = re.match(r"(\d+)'(\d+)", height_str)
            if match:
                feet, inches = map(int, match.groups())
                return (feet * 12 + inches) * 2.54
        
        # Extraer número (ignorar unidades)
        match = re.search(r"(\d+\.?\d*)", height_str)
        if match:
            return float(match.group(1))
    
    # Valor por defecto si no se puede parsear
    logger.warning(f"No se pudo parsear altura: {height_value}, usando valor por defecto 180")
    return 180.0


def parse_weight(weight_value: Any) -> float:
    """
    Convierte diferentes formatos de peso a kilogramos.
    
    Args:
        weight_value: Puede ser float, int, o string (ej: "980 lb", "445 kg", "445")
    
    Returns:
        float: Peso en kilogramos
    """
    if isinstance(weight_value, (int, float)):
        return float(weight_value)
    
    if isinstance(weight_value, str):
        weight_str = weight_value.strip().lower()
        
        # Extraer número
        match = re.search(r"(\d+\.?\d*)", weight_str)
        if match:
            value = float(match.group(1))
            
            # Convertir de libras a kg si es necesario
            if "lb" in weight_str or "pound" in weight_str:
                return value * 0.453592
            
            return value
    
    # Valor por defecto si no se puede parsear
    logger.warning(f"No se pudo parsear peso: {weight_value}, usando valor por defecto 80")
    return 80.0


def preprocess_features(features: Features) -> np.ndarray:
    """
    Aplica el preprocesamiento a las características crudas.
    
    Args:
        features: Objeto Features con las características crudas
    
    Returns:
        np.ndarray: Array preprocesado listo para predicción
    """
    # Parsear altura y peso
    height_cm = parse_height(features.height)
    weight_kg = parse_weight(features.weight_kg)
    
    # Crear array con las características
    # Orden: intelligence, strength, speed, durability, combat, height_cm, weight_kg
    features_array = np.array([
        float(features.intelligence),
        float(features.strength),
        float(features.speed),
        float(features.durability),
        float(features.combat),
        height_cm,
        weight_kg
    ]).reshape(1, -1)
    
    # Aplicar escalado si existe el scaler
    if scaler is not None:
        features_array = scaler.transform(features_array)
    
    return features_array


@app.on_event("startup")
async def load_model():
    """Carga el modelo y preprocesador al iniciar la aplicación."""
    global model, scaler, model_info

    try:
        # Rutas a los archivos
        base_path = Path(__file__).parent
        model_path = base_path / "model" / "model.pkl"

        # Verificar existencia
        if not model_path.exists():
            logger.error(f"No se encontró el archivo del modelo en {model_path}")
            raise FileNotFoundError(f"model.pkl no encontrado en {model_path}")

        # Cargar el pickle
        with open(model_path, "rb") as f:
            loaded = pickle.load(f)

        # En tu caso, model.pkl es un dict con claves:
        # ['model', 'model_type', 'best_params', 'rmse', 'feature_names', 'preprocessing']
        if isinstance(loaded, dict):
            # 👉 Aquí está el modelo de sklearn (RandomForest, etc.)
            if "model" not in loaded:
                raise ValueError(
                    f"El pickle es un dict pero no contiene la clave 'model'. "
                    f"Claves disponibles: {list(loaded.keys())}"
                )

            model = loaded["model"]

            # Si en 'preprocessing' guardaste un scaler/pipeline con .transform
            preprocessing = loaded.get("preprocessing")
            if hasattr(preprocessing, "transform"):
                scaler = preprocessing
            else:
                scaler = None

            # Información del modelo para el endpoint /info
            model_info = {
                "team_name": "Equipo GitWars",
                "model_type": loaded.get("model_type", "Modelo sklearn"),
                "hyperparameters": loaded.get("best_params", {}),
                "preprocessing": str(loaded.get("preprocessing", "Sin información específica")),
            }

        else:
            # Caso alternativo: el pickle ya es directamente un modelo sklearn
            model = loaded
            scaler = None
            model_info = {
                "team_name": "Equipo GitWars",
                "model_type": type(loaded).__name__,
                "hyperparameters": {},
                "preprocessing": "Sin información específica",
            }

        logger.info(f"Modelo cargado exitosamente desde {model_path}")

    except Exception as e:
        logger.error(f"Error al cargar el modelo: {str(e)}")
        raise



@app.get("/health")
async def health_check():
    """
    Endpoint de verificación rápida del estado del servicio.
    
    Returns:
        dict: Estado del servicio
    """
    return {"status": "ok"}


@app.get("/info")
async def get_info():
    """
    Endpoint que devuelve información sobre el equipo y el modelo.
    
    Returns:
        dict: Información del equipo, modelo e hiperparámetros
    """
    return {
        "team_name": model_info.get("team_name", "Equipo GitWars"),
        "model_type": model_info.get("model_type", "Random Forest"),
        "hyperparameters": model_info.get("hyperparameters", {}),
        "preprocessing": model_info.get("preprocessing", "StandardScaler aplicado"),
        "features": [
            "intelligence",
            "strength", 
            "speed",
            "durability",
            "combat",
            "height_cm",
            "weight_kg"
        ]
    }


@app.post("/predict", response_model=PredictResponse)
async def predict(request: PredictRequest):
    """
    Endpoint principal de inferencia.
    Recibe características crudas, aplica preprocesamiento y devuelve la predicción.
    
    Args:
        request: PredictRequest con las características del superhéroe
    
    Returns:
        PredictResponse: Predicción del poder
    
    Raises:
        HTTPException: Si hay error en la predicción
    """
    try:
        # Validar que el modelo esté cargado
        if model is None:
            raise HTTPException(
                status_code=503,
                detail="Modelo no disponible. El servicio está iniciando."
            )
        
        # Preprocesar características
        features_processed = preprocess_features(request.features)
        
        # Realizar predicción
        prediction = model.predict(features_processed)
        
        # Asegurar que la predicción es un float
        prediction_value = float(prediction[0])
        
        logger.info(f"Predicción realizada: {prediction_value}")
        
        return PredictResponse(prediction=prediction_value)
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error en predicción: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error al realizar la predicción: {str(e)}"
        )


@app.get("/")
async def root():
    """
    Endpoint raíz con información básica de la API.
    
    Returns:
        dict: Información de bienvenida
    """
    return {
        "message": "Superhero Power Prediction API",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "info": "/info",
            "predict": "/predict (POST)",
            "docs": "/docs"
        }
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)


