"""
Utilidades compartidas para el proyecto
Funciones auxiliares para carga de datos y métricas
"""

import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score


def cargar_dataset(ruta='data/data.csv'):
    """Carga el dataset desde el archivo CSV."""
    return pd.read_csv(ruta)


def calcular_rmse(y_real, y_pred):
    """Calcula el Root Mean Squared Error."""
    return np.sqrt(mean_squared_error(y_real, y_pred))


def calcular_r2(y_real, y_pred):
    """Calcula el coeficiente R²."""
    return r2_score(y_real, y_pred)


def imprimir_metricas(y_real, y_pred, nombre_modelo="Modelo"):
    """Imprime las métricas de evaluación de un modelo."""
    rmse = calcular_rmse(y_real, y_pred)
    r2 = calcular_r2(y_real, y_pred)

    print(f"\nMétricas de {nombre_modelo}:")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  R²:   {r2:.4f}")

    return {'rmse': rmse, 'r2': r2}


__all__ = [
    'cargar_dataset',
    'calcular_rmse',
    'calcular_r2',
    'imprimir_metricas',
]
