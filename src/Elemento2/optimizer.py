"""
Elemento 2: Optimización Bayesiana desde cero
Implementación manual de BO usando Gaussian Process y UCB
"""

import numpy as np
from typing import Dict, Tuple

try:
    from src.Elemento1.orchestrator import evaluar_svm, evaluar_rf, evaluar_mlp
except Exception:
    from ..Elemento1.orchestrator import evaluar_svm, evaluar_rf, evaluar_mlp


def rbf_kernel(x1, x2, longitud_escala=1.0):
    """Calcula el kernel RBF entre dos conjuntos de puntos."""
    x1 = np.atleast_2d(x1)
    x2 = np.atleast_2d(x2)

    dist_cuadrado = np.sum((x1[:, np.newaxis] - x2[np.newaxis, :]) ** 2, axis=2)
    kernel = np.exp(-dist_cuadrado / (2 * longitud_escala ** 2))
    return kernel


def ajustar_gp(X, y, longitud_escala=1.0, ruido=1e-6):
    """Ajusta un Gaussian Process a los datos observados."""
    X = np.array(X)
    y = np.array(y).flatten()

    K = rbf_kernel(X, X, longitud_escala)
    K_ruido = K + ruido * np.eye(len(K))
    K_inv = np.linalg.inv(K_ruido)
    alpha = K_inv @ y

    return {
        'K_inv': K_inv,
        'alpha': alpha,
        'X_train': X,
        'longitud_escala': longitud_escala,
        'ruido': ruido,
    }


def predecir_gp(params_gp, X_test):
    """Predice la media y desviación estándar del GP en puntos nuevos."""
    X_test = np.array(X_test)
    X_train = params_gp['X_train']
    alpha = params_gp['alpha']
    K_inv = params_gp['K_inv']
    longitud_escala = params_gp['longitud_escala']

    k_star = rbf_kernel(X_test, X_train, longitud_escala)
    medias = k_star @ alpha

    k_star_star = rbf_kernel(X_test, X_test, longitud_escala)
    varianzas = np.diag(k_star_star) - np.sum(k_star @ K_inv * k_star, axis=1)
    varianzas = np.maximum(varianzas, 1e-10)
    desviaciones = np.sqrt(varianzas)

    return medias, desviaciones


def adquisicion_ucb(params_gp, X_candidatos, kappa=2.0):
    """Calcula la función de adquisición UCB (usando LCB para minimización)."""
    X_candidatos = np.array(X_candidatos)
    medias, desviaciones = predecir_gp(params_gp, X_candidatos)
    lcb_valores = medias - kappa * desviaciones
    return lcb_valores


def optimizar_modelo(tipo_modelo: str, n_init=3, n_iter=10, semilla_aleatoria=42) -> Tuple[Dict, float]:
    """Optimiza hiperparámetros usando BO sobre una rejilla discreta.

    Devuelve (mejores_params, mejor_rmse).
    """
    np.random.seed(semilla_aleatoria)

    if tipo_modelo == 'svm':
        rejilla_params = {
            'C': [0.1, 1, 10, 100],
            'gamma': [0.001, 0.01, 0.1, 1]
        }
        func_eval = evaluar_svm
    elif tipo_modelo == 'rf':
        rejilla_params = {
            'n_estimators': [10, 20, 50, 100],
            'max_depth': [2, 4, 6, 8]
        }
        func_eval = evaluar_rf
    elif tipo_modelo == 'mlp':
        rejilla_params = {
            'hidden_layer_sizes': [(16,), (32,), (64,), (32, 16)],
            'alpha': [1e-4, 1e-3, 1e-2]
        }
        func_eval = evaluar_mlp
    else:
        raise ValueError(f"Modelo '{tipo_modelo}' no soportado")

    nombres_params = list(rejilla_params.keys())

    from itertools import product
    combinaciones_rejilla = list(product(*[rejilla_params[nombre] for nombre in nombres_params]))
    tamaño_rejilla = len(combinaciones_rejilla)

    indice_a_params = {i: dict(zip(nombres_params, combo)) for i, combo in enumerate(combinaciones_rejilla)}

    indices_evaluados = set()
    resultados = {}

    indices_disponibles = list(range(tamaño_rejilla))

    for i in range(min(n_init, tamaño_rejilla)):
        idx = np.random.choice([j for j in indices_disponibles if j not in indices_evaluados])
        params_dict = indice_a_params[idx]
        rmse = func_eval(**params_dict)
        indices_evaluados.add(idx)
        resultados[idx] = rmse

    for iteracion in range(n_iter):
        if len(indices_evaluados) >= tamaño_rejilla:
            break

        X_observado = []
        y_observado = []
        for idx in indices_evaluados:
            params = indice_a_params[idx]
            x_vec = []
            for nombre in nombres_params:
                val = params[nombre]
                if isinstance(val, tuple):
                    val = sum(val)
                x_vec.append(float(val))
            X_observado.append(x_vec)
            y_observado.append(resultados[idx])

        X_observado = np.array(X_observado)
        y_observado = np.array(y_observado)

        params_gp = ajustar_gp(X_observado, y_observado, longitud_escala=1.0, ruido=1e-6)

        indices_no_evaluados = [i for i in range(tamaño_rejilla) if i not in indices_evaluados]
        X_candidatos = []
        for idx in indices_no_evaluados:
            params = indice_a_params[idx]
            x_vec = []
            for nombre in nombres_params:
                val = params[nombre]
                if isinstance(val, tuple):
                    val = sum(val)
                x_vec.append(float(val))
            X_candidatos.append(x_vec)

        X_candidatos = np.array(X_candidatos)

        lcb_valores = adquisicion_ucb(params_gp, X_candidatos, kappa=2.0)
        mejor_candidato_idx = np.argmin(lcb_valores)
        siguiente_idx_rejilla = indices_no_evaluados[mejor_candidato_idx]
        params_dict = indice_a_params[siguiente_idx_rejilla]

        rmse = func_eval(**params_dict)
        indices_evaluados.add(siguiente_idx_rejilla)
        resultados[siguiente_idx_rejilla] = rmse

    mejor_idx_rejilla = min(resultados, key=resultados.get)
    mejores_params = indice_a_params[mejor_idx_rejilla]
    mejor_rmse_final = resultados[mejor_idx_rejilla]

    return mejores_params, float(mejor_rmse_final)


__all__ = [
    'rbf_kernel',
    'ajustar_gp',
    'predecir_gp',
    'adquisicion_ucb',
    'optimizar_modelo'
]
