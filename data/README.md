# Elemento 0: Dataset

## Descripción

Este directorio contiene el dataset utilizado para el entrenamiento y evaluación de los modelos de Machine Learning.

## Contenido

- `data.csv`: Dataset principal con características y etiquetas

## Estructura del Dataset

El dataset debe contener:
- Características numéricas para entrenamiento
- Variable objetivo (target)
- División en conjuntos de entrenamiento y prueba

## Generación

El dataset se genera ejecutando el script del Elemento 0:

```bash
python -m src.Elemento0.get_data
```

O desde el código:

```python
from src.Elemento0 import fetch_superhero_data

df = fetch_superhero_data()
```

Este script:
1. Consume la SuperHero API (https://akabab.github.io/superhero-api)
2. Extrae powerstats (intelligence, strength, speed, durability, combat, power)
3. Convierte altura a cm y peso a kg
4. Limpia y valida los datos
5. Genera exactamente 600 registros

## Formato

Archivo CSV con 8 columnas numéricas:
- `intelligence`, `strength`, `speed`, `durability`, `combat`: Estadísticas de poder
- `height_cm`: Altura en centímetros
- `weight_kg`: Peso en kilogramos
- `power`: Variable objetivo (a predecir)

Todas las columnas son numéricas, sin valores faltantes.

## Consideraciones

- Verificar ausencia de valores nulos
- Normalizar o escalar características según sea necesario
- Mantener balance de clases en problemas de clasificación
