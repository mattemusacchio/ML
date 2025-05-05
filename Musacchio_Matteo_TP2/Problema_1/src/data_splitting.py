import numpy as np
from .models import LogisticRegression
from .metrics import f1_score
import pandas as pd
from .preprocessing import preprocess_file

def train_val_split(df, test_size=0.2, random_state=42):

    """
    Divide el DataFrame en conjuntos de entrenamiento y validación sin usar sklearn.
    """
    np.random.seed(random_state)
    indices = np.random.permutation(len(df))
    split_idx = int(len(df) * (1 - test_size))
    train_indices, val_indices = indices[:split_idx], indices[split_idx:]
    train_df, val_df = df.iloc[train_indices], df.iloc[val_indices]
    return train_df, val_df

def cross_val(X_raw, y_raw, k=5, epochs=1000, lr=0.01):
    """
    Validación cruzada para regresión logística binaria, con preprocesamiento dentro de cada fold.
    
    Args:
        X_raw: DataFrame de features crudos (sin procesar).
        y_raw: Series o array de etiquetas binarias.
        preprocess_fn: función de preprocesamiento que recibe un df y devuelve (X_preprocesado, params).
        k: cantidad de folds.
        epochs: épocas de entrenamiento.
        lr: learning rate.
    
    Returns:
        best_lambda: lambda con mejor F1-score promedio.
        lambda_scores: dict de lambda -> F1-score promedio.
    """
    import matplotlib.pyplot as plt
    lambdas = np.logspace(-4, 2, 60)
    indices = np.arange(len(X_raw))
    np.random.seed(42)
    np.random.shuffle(indices)

    # Reordenar los datos
    X_raw = X_raw.iloc[indices].reset_index(drop=True)
    y_raw = y_raw.iloc[indices].reset_index(drop=True)

    # Generar folds
    fold_sizes = (len(X_raw) // k) * np.ones(k, dtype=int)
    fold_sizes[:len(X_raw) % k] += 1
    current = 0
    folds = []

    for fold_size in fold_sizes:
        start, stop = current, current + fold_size
        folds.append((X_raw.iloc[start:stop], y_raw.iloc[start:stop]))
        current = stop

    lambda_scores = {}

    for lmbd in lambdas:
        f1s = []

        for i in range(k):
            # Crear train y val
            X_val_raw, y_val = folds[i]
            X_train_raw = pd.concat([folds[j][0] for j in range(k) if j != i], axis=0)
            y_train = pd.concat([folds[j][1] for j in range(k) if j != i], axis=0)

            # Preprocesar train y aplicar a val
            df_train = X_train_raw.copy()
            df_train["Diagnosis"] = y_train.values
            X_train_proc, params = preprocess_file(df_train)
            y_train_proc = X_train_proc.pop("Diagnosis")

            df_val = X_val_raw.copy()
            df_val["Diagnosis"] = y_val.values
            X_val_proc = preprocess_file(df_val, params=params)
            y_val_proc = X_val_proc.pop("Diagnosis")

            # Entrenar y evaluar
            model = LogisticRegression(X_train_proc, y_train_proc, l2=lmbd)
            model.fit_gradient_descent(lr=lr, epochs=epochs)
            y_pred = model.predict(X_val_proc)
            f1 = f1_score(y_val_proc.values, y_pred)
            f1s.append(f1)

        lambda_scores[lmbd] = np.mean(f1s)

    best_lambda = max(lambda_scores, key=lambda_scores.get)
    print(f"\nMejor lambda encontrado: {best_lambda} con F1={lambda_scores[best_lambda]:.4f}")

    # Graficar los resultados
    plt.figure(figsize=(8, 6))
    plt.plot(lambda_scores.keys(), lambda_scores.values(), marker='o', linestyle='-', color='b')
    plt.xscale('log')
    plt.xlabel('Lambda (log scale)')
    plt.ylabel('F1 Score Promedio')
    plt.title('Validación Cruzada: F1 Score vs Lambda')
    plt.axvline(x=best_lambda, color='r', linestyle='--', label=f'Mejor Lambda: {best_lambda:.5f}')
    plt.legend()
    plt.grid(True)
    plt.show()

    return best_lambda, lambda_scores