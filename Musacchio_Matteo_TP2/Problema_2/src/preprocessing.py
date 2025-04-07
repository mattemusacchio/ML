import numpy as np
import pandas as pd

class StandardScaler:
    def __init__(self):
        self.means = None
        self.stds = None

    def fit(self, X):
        """
        Calcula la media y desviación estándar para cada feature.
        Args:
            X (pd.DataFrame): DataFrame con los datos de entrada.
        """
        self.means = X.mean()
        self.stds = X.std(ddof=0)  # ddof=0 para poblacional (igual que sklearn)

    def transform(self, X):
        """
        Aplica la normalización z-score a los datos.
        Args:
            X (pd.DataFrame): Datos a transformar.
        Returns:
            X_scaled (pd.DataFrame): Datos transformados.
        """
        if self.means is None or self.stds is None:
            raise ValueError("Primero tenés que ajustar el scaler con .fit(X)")
        return (X - self.means) / self.stds

    def fit_transform(self, X):
        """
        Ajusta el scaler y transforma los datos.
        """
        self.fit(X)
        return self.transform(X)

    def inverse_transform(self, X_scaled):
        """
        Devuelve los datos originales a partir de los normalizados.
        """
        return X_scaled * self.stds + self.means
