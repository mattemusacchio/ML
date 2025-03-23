import numpy as np 
import pandas as pd

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

def cross_val(X, y, model, folds=10, l1=0.0, l2=0.0):
    from models import LinearRegression
    """
    Realiza validación cruzada k-fold para evaluar un modelo.

    Args:
        X (pd.DataFrame): Features de entrada
        y (pd.Series): Variable objetivo
        model (str): Método de entrenamiento ('gradient', 'pseudo' o 'normal')
        folds (int): Número de folds para la validación cruzada
        l1 (float): Coeficiente de regularización L1 (Lasso)
        l2 (float): Coeficiente de regularización L2 (Ridge)

    Returns:
        dict: Diccionario con las métricas promedio de todos los folds
    """
    n = len(y)
    indices = np.random.permutation(n)
    fold_size = n // folds
    metrics_per_fold = []

    for i in range(folds):
        # Definir índices de validación y entrenamiento
        val_idx = indices[i * fold_size:(i + 1) * fold_size]
        train_idx = np.setdiff1d(indices, val_idx)

        # Separar datos
        X_train = X.iloc[train_idx]
        X_val = X.iloc[val_idx]
        y_train = y.iloc[train_idx]
        y_train_log = np.log1p(y_train)
        y_val = y.iloc[val_idx]
        y_val_log = np.log1p(y_val)

        # Entrenar modelo
        reg = LinearRegression(X_train, y_train_log, l1=l1, l2=l2)
        
        if model == 'gradient':
            reg.fit_gradient_descent()
        elif model == 'pseudo':
            reg.fit_pseudo_inverse()
        elif model == 'l2':
            reg.fit_normal_equation()
        else:
            raise ValueError("model debe ser 'gradient', 'pseudo' o 'normal'")

        # Evaluar y guardar métricas
        metrics = reg.analyze_metrics(X_val, y_val_log, print_metrics_=False)
        metrics_per_fold.append(metrics)

        # Devolver lista de MSE de cada fold
    return [fold['mse'] for fold in metrics_per_fold]

