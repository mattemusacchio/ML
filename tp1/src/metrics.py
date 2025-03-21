import numpy as np
import pandas as pd

def MSE(y_true, y_pred):
    """Calcula el Error Cuadrático Medio (ECM)."""
    return np.mean((y_true - y_pred) ** 2)

def MAE(y_true, y_pred):
    """Calcula el Error Absoluto Medio (MAE)."""
    return np.mean(np.abs(y_true - y_pred))

def RMSE(y_true, y_pred):
    """Calcula la Raíz del Error Cuadrático Medio (RMSE)."""
    return np.sqrt(MSE(y_true, y_pred))

def R2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calcula el coeficiente de determinación (R2) entre los valores reales y los valores predichos.

    Args:
        y_true (np.ndarray): Array de valores reales.
        y_pred (np.ndarray): Array de valores predichos.

    Returns|:
        float: Coeficiente de determinación.
    """
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - (ss_res / ss_tot)
