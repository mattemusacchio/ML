import numpy as np 
import pandas as pd

from metrics import MSE

def train_val_split(df, test_size=0.2, random_state=95):
    """
    Divide el DataFrame en conjuntos de entrenamiento y validación sin usar sklearn.
    """
    np.random.seed(random_state)
    indices = np.random.permutation(len(df))
    split_idx = int(len(df) * (1 - test_size))
    train_indices, val_indices = indices[:split_idx], indices[split_idx:]
    train_df, val_df = df.iloc[train_indices], df.iloc[val_indices]
    return train_df, val_df

def cross_val(X, y, model_class, folds=5, **model_params):
    """
    Realiza validación cruzada k-fold para evaluar un modelo.

    Parámetros:
    - X: np.array, features de entrada.
    - y: np.array, variable objetivo.
    - model_class: clase del modelo a evaluar (por ejemplo, LinearRegression).
    - folds: int, número de folds para la validación cruzada.
    - model_params: dict, parámetros adicionales para inicializar el modelo.

    Retorna:
    - errores: lista de errores ECM en cada fold.
    """
    X = np.array(X, dtype=np.float64)
    y = np.array(y, dtype=np.float64).reshape(-1, 1)
    
    n = len(y)
    indices = np.random.permutation(n)  # Barajar índices
    fold_size = n // folds
    errores = []

    for i in range(folds):
        # Definir conjuntos de entrenamiento y validación
        val_idx = indices[i * fold_size: (i + 1) * fold_size]
        train_idx = np.setdiff1d(indices, val_idx)

        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        # Inicializar y entrenar el modelo
        model = model_class(X_train, y_train, **model_params)
        model.fit_pseudo_inverse()  # Puedes cambiar a otro método de entrenamiento

        # Hacer predicciones
        y_pred = model.predict(X_val)

        # Calcular ECM (Error Cuadrático Medio)
        error = MSE(y_val, y_pred)
        errores.append(error)

    return errores

