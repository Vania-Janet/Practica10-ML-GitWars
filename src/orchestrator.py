import pandas as pd
import numpy as np
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


def evaluate_svm(C, gamma, data_path="../data/data.csv"):
	"""Entrena y evalúa un modelo SVR con los hiperparámetros dados.

	Retorna el RMSE en el conjunto de test.
	"""
	df = pd.read_csv(data_path)
	X = df.drop('power', axis=1)
	y = df['power']

	X_train, X_test, y_train, y_test = train_test_split(
		X, y, test_size=0.2, random_state=42
	)

	model = SVR(C=C, gamma=gamma, kernel='rbf')
	model.fit(X_train, y_train)

	y_pred = model.predict(X_test)
	rmse = np.sqrt(mean_squared_error(y_test, y_pred))

	return float(rmse)


def evaluate_rf(n_estimators, max_depth, data_path="../data/data.csv"):
	"""Entrena y evalúa un RandomForestRegressor con los hiperparámetros dados.

	Retorna el RMSE en el conjunto de test.
	"""
	df = pd.read_csv(data_path)
	X = df.drop('power', axis=1)
	y = df['power']

	X_train, X_test, y_train, y_test = train_test_split(
		X, y, test_size=0.2, random_state=42
	)

	model = RandomForestRegressor(
		n_estimators=int(n_estimators),
		max_depth=int(max_depth),
		random_state=42,
		n_jobs=-1,
	)
	model.fit(X_train, y_train)

	y_pred = model.predict(X_test)
	rmse = np.sqrt(mean_squared_error(y_test, y_pred))

	return float(rmse)


def evaluate_mlp(hidden_layer_sizes, alpha, data_path="../data/data.csv"):
	"""Entrena y evalúa un MLPRegressor con los hiperparámetros dados.

	Retorna el RMSE en el conjunto de test.
	"""
	df = pd.read_csv(data_path)
	X = df.drop('power', axis=1)
	y = df['power']

	X_train, X_test, y_train, y_test = train_test_split(
		X, y, test_size=0.2, random_state=42
	)

	model = MLPRegressor(
		hidden_layer_sizes=hidden_layer_sizes,
		alpha=alpha,
		max_iter=1000,
		random_state=42,
		early_stopping=True,
		validation_fraction=0.1,
	)
	model.fit(X_train, y_train)

	y_pred = model.predict(X_test)
	rmse = np.sqrt(mean_squared_error(y_test, y_pred))

	return float(rmse)


__all__ = [
	'evaluate_svm',
	'evaluate_rf',
	'evaluate_mlp',
]

