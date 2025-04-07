import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from IPython.display import display, Markdown
from collections import Counter
from .metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    plot_roc_curve,
    plot_precision_recall_curve
)

import seaborn as sns
import matplotlib.pyplot as plt

def evaluate(y_true, y_pred, y_proba, title="Modelo"):
    """Evalúa el desempeño de un modelo binario."""
    
    print(f"Evaluación del modelo: {title}")
    print("-" * 40)
    
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    
    print(f"Accuracy:  {acc:.3f}")
    print(f"Precision: {prec:.3f}")
    print(f"Recall:    {rec:.3f}")
    print(f"F1-score:  {f1:.3f}")
    
    # Matriz de confusión
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=[0, 1], yticklabels=[0, 1])
    plt.xlabel("Predicción")
    plt.ylabel("Real")
    plt.title(f"Matriz de Confusión - {title}")
    plt.show()
    
    # Curva ROC
    plot_roc_curve(y_true, y_proba)
    
    # Curva Precision-Recall
    plot_precision_recall_curve(y_true, y_proba)


class LogisticRegression:
    """
    Implementación de una regresión logística binaria con regularización L2.
    """
    def __init__(self, X, y, l2=0.0):
        self.features = X.columns
        self.X = X
        self.y = y
        self.l2 = l2
        self.classes = np.unique(y)
        self.n_classes = len(self.classes)
        self.coef = np.zeros((self.n_classes, len(self.features) + 1))

    
    def softmax(self, Z):
        exp_Z = np.exp(Z - np.max(Z, axis=1, keepdims=True))  # estabilidad numérica
        return exp_Z / np.sum(exp_Z, axis=1, keepdims=True)
    
    def one_hot(self, y):
        one_hot = np.zeros((len(y), self.n_classes))
        for i, c in enumerate(self.classes):
            one_hot[:, i] = (y == c).astype(float)
        return one_hot


    
    def fit_gradient_descent(self, lr=0.01, epochs=1000):
        X = np.c_[np.ones((len(self.X), 1)), np.array(self.X, dtype=np.float64)]
        m = X.shape[0]
        y = self.one_hot(self.y)

        for _ in range(epochs):
            Z = X @ self.coef.T  # (m x n_classes)
            y_pred = self.softmax(Z)
            error = y_pred - y  # (m x n_classes)

            # Gradiente
            grad = (1/m) * (error.T @ X) + self.l2 * self.coef

            # Actualización
            self.coef -= lr * grad

    
    def predict_proba(self, X):
        X = np.c_[np.ones((X.shape[0], 1)), np.array(X, dtype=np.float64)]
        return self.softmax(X @ self.coef.T)

    def predict(self, X):
        proba = self.predict_proba(X)
        return self.classes[np.argmax(proba, axis=1)]

    
    def print_coefficients(self):
        from .utils import pretty_print_df
        """Imprime los coeficientes con los nombres de las características."""
        nombres_filas = ['intercept'] + list(self.features)
        coeficientes = self.coef.flatten()

        df = pd.DataFrame({'Feature': nombres_filas, 'Coeficiente': coeficientes})
        pretty_print_df(df)

    def evaluate(self, X, y_true, print_metrics=""):
        evaluate(y_true, self.predict(X), self.predict_proba(X), title=print_metrics)

class LDA:
    def __init__(self):
        self.classes = None
        self.means = None
        self.priors = None
        self.cov_inv = None

    def fit(self, X, y):
        self.classes = np.unique(y)
        self.means = {}
        self.priors = {}
        n_features = X.shape[1]
        n_samples = X.shape[0]
        
        # Calcular la matriz de covarianza común
        Sw = np.zeros((n_features, n_features))
        for c in self.classes:
            X_c = X[y == c]
            mu_c = np.mean(X_c, axis=0)
            self.means[c] = mu_c
            self.priors[c] = len(X_c) / n_samples
            # Matriz de covarianza intra-clase (scatter)
            Sw += (X_c - mu_c).T @ (X_c - mu_c)
        
        self.cov_inv = np.linalg.inv(Sw / (n_samples - len(self.classes)))
        
    def predict(self, X):
        scores = []
        for c in self.classes:
            mu = self.means[c]
            prior = self.priors[c]
            # Score discriminante (forma cuadrática)
            score = X @ self.cov_inv @ mu - 0.5 * mu.T @ self.cov_inv @ mu + np.log(prior)
            scores.append(score)
        scores = np.vstack(scores).T
        return self.classes[np.argmax(scores, axis=1)]

    def score(self, X, y):
        y_pred = self.predict(X)
        return np.mean(y_pred == y)
    
    def evaluate(self, X, y_true, print_metrics=""):
        evaluate(y_true, self.predict(X), self.predict_proba(X), title=print_metrics)

# --- ENTROPÍA Y GANANCIA DE INFORMACIÓN ---

def entropy(y):
    """Calcula la entropía de una distribución de clases."""
    counts = np.bincount(y)
    probs = counts / len(y)
    return -np.sum([p * np.log2(p) for p in probs if p > 0])

def information_gain(y, left_idx, right_idx):
    """Calcula la ganancia de información para una división."""
    left_y, right_y = y[left_idx], y[right_idx]
    p = len(left_y) / len(y)
    return entropy(y) - p * entropy(left_y) - (1 - p) * entropy(right_y)

# --- NODO DE ÁRBOL DE DECISIÓN ---

class DecisionNode:
    def __init__(self, feature=None, threshold=None, left=None, right=None, *, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value  # Clase si es hoja

# --- ÁRBOL DE DECISIÓN ---

class DecisionTree:
    def __init__(self, max_depth=10, min_samples_split=2, n_features=None):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.n_features = n_features
        self.root = None

    def fit(self, X, y):
        self.n_classes = len(set(y))
        self.n_features = X.shape[1] if self.n_features is None else self.n_features
        self.root = self._grow_tree(X, y)

    def _grow_tree(self, X, y, depth=0):
        n_samples, n_features = X.shape
        n_labels = len(set(y))

        if (depth >= self.max_depth or n_labels == 1 or n_samples < self.min_samples_split):
            leaf_value = self._most_common_label(y)
            return DecisionNode(value=leaf_value)

        feat_idxs = np.random.choice(n_features, self.n_features, replace=False)

        # encontrar mejor split
        best_gain = -1
        split = None
        for feat in feat_idxs:
            thresholds = np.unique(X[:, feat])
            for t in thresholds:
                left_idx = np.where(X[:, feat] <= t)[0]
                right_idx = np.where(X[:, feat] > t)[0]
                if len(left_idx) == 0 or len(right_idx) == 0:
                    continue

                gain = information_gain(y, left_idx, right_idx)

                if gain > best_gain:
                    split = {
                        'feature': feat,
                        'threshold': t,
                        'left_idx': left_idx,
                        'right_idx': right_idx
                    }
                    best_gain = gain

        if split is None:
            return DecisionNode(value=self._most_common_label(y))

        left = self._grow_tree(X[split['left_idx']], y[split['left_idx']], depth + 1)
        right = self._grow_tree(X[split['right_idx']], y[split['right_idx']], depth + 1)

        return DecisionNode(feature=split['feature'], threshold=split['threshold'], left=left, right=right)

    def _most_common_label(self, y):
        counter = Counter(y)
        return counter.most_common(1)[0][0]

    def predict(self, X):
        return [self._traverse_tree(x, self.root) for x in X]

    def _traverse_tree(self, x, node):
        if node.value is not None:
            return node.value

        if x[node.feature] <= node.threshold:
            return self._traverse_tree(x, node.left)
        return self._traverse_tree(x, node.right)

# --- RANDOM FOREST CLASSIFIER ---

class RandomForest:
    def __init__(self, n_trees=10, max_depth=10, min_samples_split=2, n_features=None):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.n_features = n_features
        self.trees = []

    def fit(self, X, y):
        self.trees = []
        for _ in range(self.n_trees):
            tree = DecisionTree(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                n_features=self.n_features
            )
            X_sample, y_sample = self._bootstrap_sample(X, y)
            tree.fit(X_sample, y_sample)
            self.trees.append(tree)

    def _bootstrap_sample(self, X, y):
        n_samples = X.shape[0]
        idxs = np.random.choice(n_samples, n_samples, replace=True)
        return X[idxs], y[idxs]

    def predict(self, X):
        tree_preds = np.array([tree.predict(X) for tree in self.trees])
        tree_preds = np.swapaxes(tree_preds, 0, 1)  # [n_samples, n_trees]
        return [Counter(row).most_common(1)[0][0] for row in tree_preds]
    
    def evaluate(self, X, y_true, print_metrics=""):
        evaluate(y_true, self.predict(X), self.predict_proba(X), title=print_metrics)
