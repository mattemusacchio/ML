import numpy as np

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
        self.X = np.array(X, dtype=np.float64)
        self.y = np.array(y, dtype=np.float64).reshape(-1, 1)
        self.l1 = l1
        self.l2 = l2
        self.coef = None

    def fit_pseudo_inverse(self):
        """Entrena el modelo utilizando la pseudo-inversa."""
        # Calcular la pseudo-inversa usando la fórmula (X^T X)^(-1) X^T y
        XTX = self.X.T.dot(self.X)
        XTX_inv = np.linalg.inv(XTX)
        XT_y = self.X.T.dot(self.y)
        self.coef = XTX_inv.dot(XT_y)

    def fit_normal_equation(self):
        """Entrena el modelo usando la ecuación normal con regularización L2 (Ridge)."""
        m, n = self.X.shape
        I = np.eye(n)  # Matriz identidad
        I[0, 0] = 0  # No regularizar el término de sesgo (bias)
        
        XTX = self.X.T.dot(self.X) + self.l2 * I  # Agregar regularización L2
        XTX_inv = np.linalg.inv(XTX)
        XT_y = self.X.T.dot(self.y)
        self.coef = XTX_inv.dot(XT_y)

    def fit_gradient_descent(self, lr=0.01, epochs=1000):
        """Entrena el modelo usando descenso por gradiente con regularización L1 y L2."""
        m, n = self.X.shape
        self.coef = np.zeros((n, 1))

        for _ in range(epochs):
            gradient = (1/m) * self.X.T.dot(self.X.dot(self.coef) - self.y)
            
            # Regularización L2 (Ridge)
            gradient += self.l2 * self.coef  
            
            # Regularización L1 (Lasso)
            gradient += self.l1 * np.sign(self.coef)  

            self.coef -= lr * gradient

    def predict(self, X):
        """Realiza predicciones para nuevos datos."""
        X = np.array(X, dtype=np.float64)
        return X.dot(self.coef)

    def print_coefficients(self, feature_names):
        """Imprime los coeficientes con nombres de variables."""
        coef_dict = {name: coef for name, coef in zip(feature_names, self.coef.flatten())}
        for key, value in coef_dict.items():
            print(f"{key}: {value}")
