"""
Elemento 2: Búsqueda aleatoria de hiperparámetros
Método de referencia para comparar con Optimización Bayesiana
"""

import numpy as np
from typing import Dict, Tuple

try:
    from src.Elemento1.orchestrator import evaluar_svm, evaluar_rf, evaluar_mlp
except Exception:
    from ..Elemento1.orchestrator import evaluar_svm, evaluar_rf, evaluar_mlp


def busqueda_aleatoria(tipo_modelo: str, n_iter: int = 13, semilla_aleatoria: int = 42) -> Tuple[Dict, float]:
    """Búsqueda aleatoria sobre una rejilla discreta de hiperparámetros.

    Retorna (mejores_params, mejor_rmse).
    """
    np.random.seed(semilla_aleatoria)

    espacios = {
        "svm": {
            "C": [0.1, 1, 10, 100],
            "gamma": [0.001, 0.01, 0.1, 1],
        },
        "rf": {
            "n_estimators": [10, 20, 50, 100],
            "max_depth": [2, 4, 6, 8],
        },
        "mlp": {
            "hidden_layer_sizes": [(16,), (32,), (64,), (32, 16)],
            "alpha": [1e-4, 1e-3, 1e-2],
        },
    }

    funcs_eval = {
        "svm": evaluar_svm,
        "rf": evaluar_rf,
        "mlp": evaluar_mlp,
    }

    if tipo_modelo not in espacios:
        raise ValueError(f"Modelo '{tipo_modelo}' no soportado")

    rejilla_params = espacios[tipo_modelo]
    func_eval = funcs_eval[tipo_modelo]
    nombres_params = list(rejilla_params.keys())

    from itertools import product
    rejilla = list(product(*[rejilla_params[nombre] for nombre in nombres_params]))
    n_total = len(rejilla)
    n_eval = min(n_iter, n_total)

    indices = np.random.choice(n_total, size=n_eval, replace=False)

    mejor_rmse = np.inf
    mejores_params = None

    for idx in indices:
        params = dict(zip(nombres_params, rejilla[idx]))
        rmse = func_eval(**params)

        if rmse < mejor_rmse:
            mejor_rmse = rmse
            mejores_params = params

    return mejores_params, float(mejor_rmse)


__all__ = ['busqueda_aleatoria']
