"""
Elemento 1: Orquestador de evaluación de modelos
Funciones para entrenar y evaluar SVM, Random Forest y MLP
"""

import pandas as pd
import numpy as np
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


def evaluar_svm(C, gamma, ruta_datos="../data/data.csv"):
    """Entrena y evalúa un modelo SVR con los hiperparámetros dados.

    Retorna el RMSE en el conjunto de test.
    """
    df = pd.read_csv(ruta_datos)
    X = df.drop('power', axis=1)
    y = df['power']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    modelo = SVR(C=C, gamma=gamma, kernel='rbf')
    modelo.fit(X_train, y_train)

    y_pred = modelo.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    return float(rmse)


def evaluar_rf(n_estimators, max_depth, ruta_datos="../data/data.csv"):
    """Entrena y evalúa un RandomForestRegressor con los hiperparámetros dados.

    Retorna el RMSE en el conjunto de test.
    """
    df = pd.read_csv(ruta_datos)
    X = df.drop('power', axis=1)
    y = df['power']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    modelo = RandomForestRegressor(
        n_estimators=int(n_estimators),
        max_depth=int(max_depth),
        random_state=42,
        n_jobs=-1,
    )
    modelo.fit(X_train, y_train)

    y_pred = modelo.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    return float(rmse)


def evaluar_mlp(hidden_layer_sizes, alpha, ruta_datos="../data/data.csv"):
    """Entrena y evalúa un MLPRegressor con los hiperparámetros dados.

    Retorna el RMSE en el conjunto de test.
    """
    df = pd.read_csv(ruta_datos)
    X = df.drop('power', axis=1)
    y = df['power']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    modelo = MLPRegressor(
        hidden_layer_sizes=hidden_layer_sizes,
        alpha=alpha,
        max_iter=1000,
        random_state=42,
        early_stopping=True,
        validation_fraction=0.1,
    )
    modelo.fit(X_train, y_train)

    y_pred = modelo.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    return float(rmse)


__all__ = [
    'evaluar_svm',
    'evaluar_rf',
    'evaluar_mlp',
]
