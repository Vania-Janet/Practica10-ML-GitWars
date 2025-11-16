import numpy as np
from typing import Dict, Tuple

# Try to import evaluation functions from orchestrator. The import style is tolerant
try:
	from orchestrator import evaluate_svm, evaluate_rf, evaluate_mlp
except Exception:
	# Fallback in case this module is used as a package
	from .orchestrator import evaluate_svm, evaluate_rf, evaluate_mlp  # type: ignore


def rbf_kernel(x1, x2, length_scale=1.0):
	"""Calcula el kernel RBF entre dos conjuntos de puntos."""
	x1 = np.atleast_2d(x1)
	x2 = np.atleast_2d(x2)

	dist_sq = np.sum((x1[:, np.newaxis] - x2[np.newaxis, :]) ** 2, axis=2)
	kernel = np.exp(-dist_sq / (2 * length_scale ** 2))
	return kernel


def fit_gp(X, y, length_scale=1.0, noise=1e-6):
	X = np.array(X)
	y = np.array(y).flatten()

	K = rbf_kernel(X, X, length_scale)
	K_noise = K + noise * np.eye(len(K))
	K_inv = np.linalg.inv(K_noise)
	alpha = K_inv @ y

	return {
		'K_inv': K_inv,
		'alpha': alpha,
		'X_train': X,
		'length_scale': length_scale,
		'noise': noise,
	}


def gp_predict(gp_params, X_test):
	X_test = np.array(X_test)
	X_train = gp_params['X_train']
	alpha = gp_params['alpha']
	K_inv = gp_params['K_inv']
	length_scale = gp_params['length_scale']

	k_star = rbf_kernel(X_test, X_train, length_scale)
	means = k_star @ alpha

	k_star_star = rbf_kernel(X_test, X_test, length_scale)
	variances = np.diag(k_star_star) - np.sum(k_star @ K_inv * k_star, axis=1)
	variances = np.maximum(variances, 1e-10)
	stds = np.sqrt(variances)

	return means, stds


def acquisition_ucb(gp_params, X_candidates, kappa=2.0):
	X_candidates = np.array(X_candidates)
	means, stds = gp_predict(gp_params, X_candidates)
	lcb_values = means - kappa * stds
	return lcb_values


def optimize_model(model_type: str, n_init=3, n_iter=10, random_state=42) -> Tuple[Dict, float]:
	"""Optimiza hiperparámetros usando una BO sobre una rejilla discreta.

	Devuelve (best_params, best_rmse).
	"""
	np.random.seed(random_state)

	if model_type == 'svm':
		param_grid = {
			'C': [0.1, 1, 10, 100],
			'gamma': [0.001, 0.01, 0.1, 1]
		}
		eval_func = evaluate_svm
	elif model_type == 'rf':
		param_grid = {
			'n_estimators': [10, 20, 50, 100],
			'max_depth': [2, 4, 6, 8]
		}
		eval_func = evaluate_rf
	elif model_type == 'mlp':
		param_grid = {
			'hidden_layer_sizes': [(16,), (32,), (64,), (32, 16)],
			'alpha': [1e-4, 1e-3, 1e-2]
		}
		eval_func = evaluate_mlp
	else:
		raise ValueError(f"Modelo '{model_type}' no soportado")

	param_names = list(param_grid.keys())

	from itertools import product
	grid_combinations = list(product(*[param_grid[name] for name in param_names]))
	grid_size = len(grid_combinations)

	idx_to_params = {i: dict(zip(param_names, combo)) for i, combo in enumerate(grid_combinations)}

	evaluated_indices = set()
	results = {}

	available_indices = list(range(grid_size))

	# Inicialización aleatoria
	for i in range(min(n_init, grid_size)):
		idx = np.random.choice([j for j in available_indices if j not in evaluated_indices])
		params_dict = idx_to_params[idx]
		rmse = eval_func(**params_dict)
		evaluated_indices.add(idx)
		results[idx] = rmse

	# Fase de optimización
	for iteration in range(n_iter):
		if len(evaluated_indices) >= grid_size:
			break

		X_observed = []
		y_observed = []
		for idx in evaluated_indices:
			params = idx_to_params[idx]
			x_vec = []
			for name in param_names:
				val = params[name]
				if isinstance(val, tuple):
					val = sum(val)
				x_vec.append(float(val))
			X_observed.append(x_vec)
			y_observed.append(results[idx])

		X_observed = np.array(X_observed)
		y_observed = np.array(y_observed)

		gp_params = fit_gp(X_observed, y_observed, length_scale=1.0, noise=1e-6)

		unevaluated_indices = [i for i in range(grid_size) if i not in evaluated_indices]
		X_candidates = []
		for idx in unevaluated_indices:
			params = idx_to_params[idx]
			x_vec = []
			for name in param_names:
				val = params[name]
				if isinstance(val, tuple):
					val = sum(val)
				x_vec.append(float(val))
			X_candidates.append(x_vec)

		X_candidates = np.array(X_candidates)

		lcb_values = acquisition_ucb(gp_params, X_candidates, kappa=2.0)
		best_candidate_idx = np.argmin(lcb_values)
		next_grid_idx = unevaluated_indices[best_candidate_idx]
		params_dict = idx_to_params[next_grid_idx]

		rmse = eval_func(**params_dict)
		evaluated_indices.add(next_grid_idx)
		results[next_grid_idx] = rmse

	best_grid_idx = min(results, key=results.get)
	best_params = idx_to_params[best_grid_idx]
	best_rmse_final = results[best_grid_idx]

	return best_params, float(best_rmse_final)


__all__ = [
	'rbf_kernel', 'fit_gp', 'gp_predict', 'acquisition_ucb', 'optimize_model'
]

