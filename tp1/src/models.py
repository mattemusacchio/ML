import numpy as np
from utils import pretty_print_df
import pandas as pd
from metrics import MSE, MAE, RMSE, R2

class LinearRegression:
    """
    Implementación de una regresión lineal con:
    - Pseudo-inversa con regularización L2.
    - Descenso por gradiente con regularización L1 y L2.
    """
    def __init__(self, X, y, l1=0.0, l2=0.0):
        """
        Inicializa el modelo de regresión lineal con regularización.

        Args:
            X: Features de entrada.
            y: Variable objetivo.
            l1: Coeficiente de regularización L1 (Lasso).
            l2: Coeficiente de regularización L2 (Ridge).
        """
        # Agregar columna de 1's para el intercepto
        self.features = X.columns
        self.X = X
        self.y = y
        self.betas = np.zeros(len(self.features) + 1)
        self.l1 = l1
        self.l2 = l2
        self.coef = None

    def compute_loss(self, X: pd.DataFrame = None, Y: pd.Series = None, metrics=['mse'], print_text=True):
        """
        Calcula el costo según las métricas especificadas.

        Args:
            X (pd.DataFrame): Los datos de prueba. Si no se pasa, se utilizan los datos de entrenamiento.
            Y (pd.Series): El vector de etiquetas. Si no se pasa, se utilizan las etiquetas de entrenamiento.
            metrics (str o list): Métrica(s) a calcular. Opciones: 'mse', 'mae', 'rmse', 'r2', 'all'. Puede ser lista o string.
            print_text (bool): Si es True, imprime el texto con el costo del modelo.

        Returns:
            dict o print: Diccionario con las métricas y sus valores, o print con la información formateada.
        """
        predictions = self.predict(self.X) if X is None else self.predict(X)
        ground_truth = self.y if Y is None else Y

        # Definimos las métricas disponibles
        available_metrics = {
            'mse': MSE(ground_truth, predictions),
            'mae': MAE(ground_truth, predictions),
            'rmse': RMSE(ground_truth, predictions),
            'r2': R2(ground_truth, predictions)
        }

        # Si el usuario pasa solo un string, lo convertimos en una lista
        if isinstance(metrics, str):
            if metrics.lower() == 'all':
                metrics = list(available_metrics.keys())
            else:
                metrics = [metrics.lower()]
        else:
            metrics = [metric.lower() for metric in metrics]


        # Filtrar solo las métricas solicitadas
        selected_metrics = {key: available_metrics[key] for key in metrics if key in available_metrics}

        # Si no hay métricas válidas, lanzamos error
        if not selected_metrics:
            raise ValueError("Invalid metric. Use 'mse', 'mae', 'rmse', 'r2', 'all' or a list with multiple metrics.")

        # Retorno en formato diccionario o string formateado
        if print_text:
            print("Model's loss:")
            print(', '.join([f"{key.upper()}: {value:,.6f}" for key, value in selected_metrics.items()]))
            return
        
        return selected_metrics

    def fit_pseudo_inverse(self):
        """Entrena el modelo utilizando la pseudo-inversa."""
        X = np.c_[np.ones((len(self.X), 1)), self.X]
        self.coef = np.linalg.pinv(X) @ self.y
        return self

    def fit_normal_equation(self):
        """Entrena el modelo usando la ecuación normal con regularización L2 (Ridge)."""
        X = np.c_[np.ones((len(self.X), 1)), self.X]
        n = X.shape[1]
        I = np.eye(n)
        I[0, 0] = 0  # No regularizar el intercepto
        
        self.coef = np.linalg.inv(X.T @ X + self.l2 * I) @ X.T @ self.y
        return self

    def fit_gradient_descent(self, lr=0.01, epochs=1000):
        """Entrena el modelo usando descenso por gradiente con regularización L1 y L2."""
        X = np.c_[np.ones((len(self.X), 1)), self.X]
        m = X.shape[0]
        self.coef = np.zeros(X.shape[1])
        
        for _ in range(epochs):
            y_pred = X @ self.coef
            error = y_pred - self.y
            
            gradient = (1/m) * X.T @ error
            gradient += self.l2 * self.coef  # L2
            gradient += self.l1 * np.sign(self.coef)  # L1
            
            self.coef -= lr * gradient
            
        return self

    def predict(self, X):
        """Realiza predicciones para nuevos datos."""
        # Agregar columna de 1's para el intercepto
        X = np.c_[np.ones((X.shape[0], 1)), np.array(X, dtype=np.float64)]
        return X.dot(self.coef)

    def print_coefficients(self):
        """Imprime los coeficientes con nombres de variables."""
        df = pd.DataFrame(self.coef.flatten(), index=['intercept'] + list(self.features), columns=['Coeficiente'])
        pretty_print_df(df, title="Coeficientes del modelo")
