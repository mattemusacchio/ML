import numpy as np
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
        self.l1 = l1
        self.l2 = l2
        # Inicializar coef como array de numpy
        self.coef = np.zeros(len(self.features) + 1)  # +1 para el intercepto

    def analyze_metrics(self, X: pd.DataFrame = None, Y: pd.Series = None, print_metrics_=True):
        """Calcula las métricas de evaluación del modelo.

        Args:
            X (pd.DataFrame, opcional): Datos de entrada para evaluar. Si es None, usa los datos de entrenamiento.
            Y (pd.Series, opcional): Valores objetivo para evaluar. Si es None, usa los valores de entrenamiento.
            print_text (bool, opcional): Si es True imprime las métricas formateadas. Si es False retorna un diccionario.

        Returns:
            dict o None: Si print_text es False retorna un diccionario con las métricas calculadas.
                        Si print_text es True imprime las métricas y retorna None.
        """
        predict_log = self.predict(X)
        predict = np.expm1(predict_log)
        real_log = Y
        real = np.expm1(real_log)

        metrics = {
            'mse': MSE(real, predict),
            'mae': MAE(real, predict),
            'rmse': RMSE(real, predict),
            'r2': R2(real, predict)
        }

        if print_metrics_:
            from utils import print_metrics
            print_metrics(real, predict)
            return
        
        return metrics

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
        X = np.c_[np.ones((len(self.X), 1)), np.array(self.X, dtype=np.float64)]
        m = X.shape[0]
        
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
        from utils import pretty_print_df
        """Imprime los coeficientes con nombres de variables."""
        # Aseguramos que cada fila tenga el nombre de la feature correspondiente
        nombres_filas = ['intercept'] + list(self.features)
        coeficientes = self.coef.flatten()
        
        # Creamos un DataFrame sin índice para que los nombres aparezcan como una columna
        data = {'Feature': nombres_filas, 'Coeficiente': coeficientes}
        df = pd.DataFrame(data)
        
        pretty_print_df(df, title="Coeficientes del modelo")
