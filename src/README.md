# Elementos 1, 2 y 3: Código Fuente Principal

## Descripción

Este directorio contiene la implementación de los evaluadores de modelos, algoritmos de optimización y funciones auxiliares.

## Archivos

### orchestrator.py - Elemento 1
Implementa los evaluadores para tres tipos de modelos:
- **SVM (Support Vector Machine)**: Clasificación con kernel configurables
- **Random Forest**: Ensemble de árboles de decisión
- **MLP (Multi-Layer Perceptron)**: Red neuronal feedforward

Funciones principales:
- Configuración de hiperparámetros
- Entrenamiento de modelos
- Evaluación con métricas estándar

### optimizer.py - Elemento 2
Implementa el algoritmo de Optimización Bayesiana:
- **Gaussian Process (GP)**: Modelo probabilístico del espacio de hiperparámetros
- **Upper Confidence Bound (UCB)**: Función de adquisición para exploración/explotación

Características:
- Búsqueda eficiente de hiperparámetros óptimos
- Balance entre exploración y explotación
- Reducción de evaluaciones necesarias

### random_search.py - Elemento 3
Implementa búsqueda aleatoria para comparación:
- Muestreo aleatorio del espacio de hiperparámetros
- Evaluación de múltiples configuraciones
- Métrica de referencia (baseline)

### utils.py
Funciones auxiliares compartidas:
- Carga y preprocesamiento de datos
- Cálculo de métricas (accuracy, precision, recall, F1)
- Visualización de resultados
- Gestión de experimentos

## Uso

```python
# Ejemplo de uso del orchestrator
from src.orchestrator import evaluate_svm, evaluate_rf, evaluate_mlp

# Ejemplo de optimización bayesiana
from src.optimizer import bayesian_optimization

# Ejemplo de búsqueda aleatoria
from src.random_search import random_search_optimization
```

## Dependencias

- scikit-learn
- NumPy
- scipy
- GPy o scikit-optimize
