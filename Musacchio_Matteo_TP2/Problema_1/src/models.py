import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from IPython.display import display, Markdown
from .metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    plot_roc_curve,
    plot_precision_recall_curve
)

class LogisticRegression:
    """
    Implementación de una regresión logística binaria con regularización L2.
    """
    def __init__(self, X, y, l2=0.0):
        """
        Inicializa el modelo de regresión logística con regularización L2.

        Args:
            X: Features de entrada.
            y: Variable objetivo binaria.
            l2: Coeficiente de regularización L2 (Ridge).
        """
        self.features = X.columns
        self.X = X
        self.y = y
        self.l2 = l2
        self.coef = np.zeros(len(self.features) + 1)  # +1 para el intercepto
    
    def sigmoid(self, z):
        """Función sigmoide."""
        return 1 / (1 + np.exp(-z))
    
    def fit_gradient_descent(self, lr=0.01, epochs=1000, reweight=False):
        """Entrena el modelo usando descenso por gradiente con regularización L2."""
        X = np.c_[np.ones((len(self.X), 1)), np.array(self.X, dtype=np.float64)]
        m = X.shape[0]
        
        for _ in range(epochs):
            y_pred = self.sigmoid(X @ self.coef)
            self.y = np.array(self.y).ravel()

            if not reweight:
                error = y_pred - self.y
                gradient = (1/m) * X.T @ error
            else:
                # Pesos para las clases
                pi_1 = np.mean(self.y == 1)
                pi_0 = np.mean(self.y == 0)
                C = pi_0 / pi_1  # Cost re-weighting

                # Creamos vector de pesos (1 para clase 0, C para clase 1)
                sample_weights = np.where(self.y == 1, C, 1)

                # Error ponderado
                error = (y_pred - self.y) * sample_weights

                # Gradiente ponderado
                gradient = (1/m) * X.T @ error

            gradient += self.l2 * self.coef  # Regularización L2
            
            self.coef -= lr * gradient
        
        return self
    
    def predict_proba(self, X):
        """Calcula las probabilidades de pertenecer a la clase positiva."""
        X = np.c_[np.ones((X.shape[0], 1)), np.array(X, dtype=np.float64)]
        return self.sigmoid(X.dot(self.coef))
    
    def predict(self, X, threshold=0.5):
        """Genera predicciones binarias basado en un umbral."""
        return (self.predict_proba(X) >= threshold).astype(int)
    
    def print_coefficients(self):
        from .utils import pretty_print_df
        """Imprime los coeficientes con los nombres de las características."""
        nombres_filas = ['intercept'] + list(self.features)
        coeficientes = self.coef.flatten()

        df = pd.DataFrame({'Feature': nombres_filas, 'Coeficiente': coeficientes})
        pretty_print_df(df)

    def evaluate(self, X, y_true,print_metrics=""):
        from .utils import pretty_print_df
        """Calcula métricas de evaluación."""

        y_pred = self.predict(X)
        y_proba = self.predict_proba(X)

        y_true = np.array(y_true).flatten()
        y_pred = np.array(y_pred).flatten()
        y_proba = np.array(y_proba).flatten()

        if y_proba.ndim > 1:
            y_proba = y_proba[:, 1] 

        tpr_list, fpr_list = roc_auc_score(y_true, y_proba)
        precision_list, recall_list = average_precision_score(y_true, y_proba)

        metrics = {
            "Matriz de Confusión": confusion_matrix(y_true, y_pred),
            "Accuracy": accuracy_score(y_true, y_pred),
            "Precision": precision_score(y_true, y_pred),
            "Recall": recall_score(y_true, y_pred),
            "F1-Score": f1_score(y_true, y_pred),
            "AUC-ROC": np.trapz(tpr_list, fpr_list),
            "AUC-PR": np.trapz(precision_list, recall_list),
        }

        if print_metrics != "":
            # Mostramos las métricas del modelo en el dataset de test
            pd_metrics = pd.DataFrame({
                'Métrica': ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC-ROC', 'AUC-PR'],
                'Valor': [
                    metrics['Accuracy'],
                    metrics['Precision'],
                    metrics['Recall'],
                    metrics['F1-Score'],
                    metrics['AUC-ROC'],
                    metrics['AUC-PR']
                ]
            })

            pretty_print_df(pd_metrics, title=f"Métricas del modelo en el dataset de {print_metrics}")

            # Mostramos la matriz de confusión en el dataset de test
            display(Markdown("#### Matriz de Confusión en el dataset de test"))
            conf_matrix = pd.DataFrame(
                metrics['Matriz de Confusión'],
                columns=['Predicción Negativa', 'Predicción Positiva'],
                index=['Real Negativa', 'Real Positiva']
            )

            pretty_print_df(conf_matrix, title=f"Matriz de confusión en el dataset de {print_metrics}",index=True)

        return metrics

    def plot_curves(self, X, y_true):
        """Grafica las curvas ROC y Precision-Recall."""

        y_proba = self.predict_proba(X)

        y_true = np.array(y_true).flatten()
        y_proba = np.array(y_proba).flatten()

        plot_roc_curve(y_true, y_proba)
        plot_precision_recall_curve(y_true, y_proba)
