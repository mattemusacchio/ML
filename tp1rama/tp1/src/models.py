import numpy as np
import pandas as pd

def gradient_descent(X, y, betas, learning_rate, epochs):
    X = np.c_[np.ones(len(X)), X]
    m = len(y)
    for _ in range(epochs):
        predictions = X @ betas
        errors = predictions - y
        gradient = (X.T @ errors) / m
        betas -= learning_rate * gradient
    return betas

class LinearRegression:
    def __init__(self, X: pd.DataFrame, y: pd.Series):
        """Esta clase implementa un modelo de regresión lineal.

        Args:
            X (pd.DataFrame): Los datos de entrenamiento normalizados.
            y (pd.Series): El vector de etiquetas.
        """
        self.X = X
        self.y = y
        self.features = self.X.columns
        self.betas = np.zeros(len(self.features) + 1)
        
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
        from metrics import mse, mae, rmse, r2
        predictions = self.predict(self.X) if X is None else self.predict(X)
        ground_truth = self.y if Y is None else Y

        # Definimos las métricas disponibles
        available_metrics = {
            'mse': mse(ground_truth, predictions),
            'mae': mae(ground_truth, predictions),
            'rmse': rmse(ground_truth, predictions),
            'r2': r2(ground_truth, predictions)
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
            
    def fit_gradient_descent(self, learning_rate=0.01, epochs=1000):
        """Entrena el modelo utilizando el método de descenso de gradiente.

        Args:
            learning_rate (float, optional): Tasa de aprendizaje. Defaults to 0.01.
            epochs (int, optional): Número de iteraciones. Defaults to 1000.

        Returns:
            np.array: Vector de coeficientes.
        """
        self.betas = gradient_descent(self.X, self.y, self.betas, learning_rate, epochs)
        print(f"Model trained with {epochs} epochs and learning rate of {learning_rate}.\nModel's performance:")
        metrics = self.compute_loss(print_text=False, metrics='all')
        print(', '.join([f"{key.upper()}: {value:,.6f}" for key, value in metrics.items()]))
        return self.betas
    
    def fit_pseudo_inverse(self):
        """Entrena el modelo utilizando el método de la pseudo-inversa.

        Returns:
            np.array: Vector de coeficientes.
        """
        X = np.c_[np.ones(len(self.X)), self.X]
        self.betas = np.linalg.pinv(X.T @ X) @ X.T @ self.y
        print("Model trained using the pseudo-inverse method.\nModel's performance:")
        metrics = self.compute_loss(print_text=False, metrics='all')
        print(', '.join([f"{key.upper()}: {value:,.6f}" for key, value in metrics.items()]))
        return self.betas
    
    def predict(self, X: pd.DataFrame):
        """Realiza predicciones utilizando el modelo entrenado.

        Args:
            X (pd.DataFrame): Los datos de prueba.

        Returns:
            np.array: Vector de predicciones.
        """
        if X is None: x = self.X
        X = np.c_[np.ones(len(X)), X]
        
        ans = X @ self.betas 
        return ans if len(ans) > 1 else ans[0]
    
    def getWeights(self):
        df = pd.DataFrame(self.betas, index=['intercept'] + list(self.features), columns=['weights'])
        return df
    
    def print_coefficients(self):
        from utils import pretty_print_df
        if self.betas is None:
            raise ValueError('The model has not been trained yet.')
        
        df = self.getWeights().T
        df.columns = [col.replace('_', ' ').title() for col in df.columns]
        pretty_print_df(df)
        return
            
    def plot_coefficients(self):
        df = self.getWeights()
        ax = df.plot(kind='bar', title='Model Coefficients', legend=False)
        ax.set_ylabel('Component weights')
        ax.set_xticklabels([label.replace('_', ' ').title() for label in df.index], rotation=45, ha='right')
        ax.grid(alpha=0.6)
    
    def __repr__(self):
        return f'LinearRegression({self.betas})'
    
    def __str__(self):
        return f'LinearRegression model with betas: {self.betas}'