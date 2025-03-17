import numpy as np

def MSE(y_true, y_pred):
    """
    Calcula el Error Cuadrático Medio (ECM) entre los valores reales y las predicciones.

    Parámetros:
    - y_true: np.array, valores reales.
    - y_pred: np.array, valores predichos.

    Retorna:
    - ECM: float, error cuadrático medio.
    """
    return np.mean((y_true - y_pred) ** 2)

def MAE(y_true, y_pred):
    """Calcula el Error Absoluto Medio (MAE)."""
    return np.mean(np.abs(y_true - y_pred))

def RMSE(y_true, y_pred):
    """Calcula la Raíz del Error Cuadrático Medio (RMSE)."""
    return np.sqrt(MSE(y_true, y_pred))

def R2(y_true, y_pred):
    """Calcula el Coeficiente de Determinación (R2)."""
    return 1 - (MSE(y_true, y_pred) / np.var(y_true))
