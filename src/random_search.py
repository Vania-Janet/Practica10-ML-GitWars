import numpy as np
from typing import Dict, Tuple

try:
	from orchestrator import evaluate_svm, evaluate_rf, evaluate_mlp
except Exception:
	from .orchestrator import evaluate_svm, evaluate_rf, evaluate_mlp  # type: ignore


def random_search(model_type: str, n_iter: int = 13, random_state: int = 42) -> Tuple[Dict, float]:
	"""Búsqueda aleatoria sobre una rejilla discreta de hiperparámetros.

	Retorna (best_params, best_rmse).
	"""
	np.random.seed(random_state)

	spaces = {
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

	eval_funcs = {
		"svm": evaluate_svm,
		"rf": evaluate_rf,
		"mlp": evaluate_mlp,
	}

	if model_type not in spaces:
		raise ValueError(f"Modelo '{model_type}' no soportado")

	param_grid = spaces[model_type]
	eval_func = eval_funcs[model_type]
	param_names = list(param_grid.keys())

	from itertools import product
	grid = list(product(*[param_grid[name] for name in param_names]))
	n_total = len(grid)
	n_eval = min(n_iter, n_total)

	indices = np.random.choice(n_total, size=n_eval, replace=False)

	best_rmse = np.inf
	best_params = None

	for idx in indices:
		params = dict(zip(param_names, grid[idx]))
		rmse = eval_func(**params)

		if rmse < best_rmse:
			best_rmse = rmse
			best_params = params

	return best_params, float(best_rmse)


__all__ = ['random_search']

