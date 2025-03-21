import numpy as np

def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calcula el Error Cuadrático Medio (MSE) entre los valores reales y los valores predichos.

    Args:
        y_true (np.ndarray): Array de valores reales.
        y_pred (np.ndarray): Array de valores predichos.

    Returns:
        float: Error Cuadrático Medio.
    """
    return np.mean((y_true - y_pred) ** 2)

def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calcula el Error Absoluto Medio (MAE) entre los valores reales y los valores predichos.

    Args:
        y_true (np.ndarray): Array de valores reales.
        y_pred (np.ndarray): Array de valores predichos.

    Returns:
        float: Error Absoluto Medio.
    """
    return np.mean(np.abs(y_true - y_pred))

def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calcula la Raíz del Error Cuadrático Medio (RMSE) entre los valores reales y los valores predichos.

    Args:
        y_true (np.ndarray): Array de valores reales.
        y_pred (np.ndarray): Array de valores predichos.

    Returns:
        float: Raíz del Error Cuadrático Medio.
    """
    return np.sqrt(mse(y_true, y_pred))

def r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
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