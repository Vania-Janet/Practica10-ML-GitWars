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

El dataset puede ser generado mediante:
1. Carga de datos externos
2. Generación sintética
3. Procesamiento de datos crudos

## Formato

Archivo CSV con columnas separadas por comas, incluyendo encabezados descriptivos.

## Consideraciones

- Verificar ausencia de valores nulos
- Normalizar o escalar características según sea necesario
- Mantener balance de clases en problemas de clasificación
