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
    plot_precision_recall_curve,
    roc_auc_score,
    average_precision_score,
    average_precision_score_rf,
)
from .utils import pretty_print_df

import seaborn as sns
import matplotlib.pyplot as plt

def evaluate(y_true, y_pred, y_proba, title="Modelo",graficar=False,rf=False):
    tpr_list, fpr_list, aucs = roc_auc_score(y_true, y_proba)
    if rf:
        precision_list, recall_list, auc_pr = average_precision_score_rf(y_true, y_proba)
    else:
        precision_list, recall_list, auc_pr = average_precision_score(y_true, y_proba)

    metrics = {
            "Accuracy": accuracy_score(y_true, y_pred),
            "Precision": precision_score(y_true, y_pred),
            "Recall": recall_score(y_true, y_pred),
            "F1-Score": f1_score(y_true, y_pred),
            "AUC-ROC": aucs,
            "AUC-PR": auc_pr,
        }
    if title == "":
        return metrics
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
    pretty_print_df(pd_metrics, title=f"Métricas del modelo en el dataset de {title}")
    
    # Matriz de confusión
    cm = confusion_matrix(y_true, y_pred)

    if graficar:
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=[1, 2,3], yticklabels=[1, 2,3])
        plt.xlabel("Predicción")
        plt.ylabel("Real")
        plt.title(f"Matriz de Confusión - {title}")
        plt.show()
        
        # Curva ROC
        plot_roc_curve(y_true, y_proba)
        
        # Curva Precision-Recall
        plot_precision_recall_curve(y_true, y_proba,rf=rf)
    return metrics


class MulticlassLogisticRegression:
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

    def evaluate(self, X, y_true, print_metrics="",graficar=False):
        return evaluate(y_true, self.predict(X), self.predict_proba(X), title=print_metrics,graficar=graficar)

class LDA:
    def __init__(self):
        self.classes = None
        self.means = {}
        self.priors = {}
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
        X = X.to_numpy() if isinstance(X, pd.DataFrame) else X
        scores = []
        for c in self.classes:
            mu = self.means[c]
            prior = self.priors[c]
            # Score discriminante (forma cuadrática)
            score = X @ self.cov_inv @ mu - 0.5 * mu.T @ self.cov_inv @ mu + np.log(prior)
            scores.append(score)
        scores = np.vstack(scores).T
        return self.classes[np.argmax(scores, axis=1)]
    
    def predict_proba(self, X):
        X = X.to_numpy() if isinstance(X, pd.DataFrame) else X
        scores = []
        for c in self.classes:
            mu = self.means[c]
            prior = self.priors[c]
            # Score discriminante (forma cuadrática)
            score = X @ self.cov_inv @ mu - 0.5 * mu.T @ self.cov_inv @ mu + np.log(prior)
            scores.append(score)
        scores = np.vstack(scores).T
        return np.exp(scores) / np.sum(np.exp(scores), axis=1, keepdims=True)

    def score(self, X, y):
        y_pred = self.predict(X)
        return np.mean(y_pred == y)
    
    def evaluate(self, X, y_true, print_metrics="",graficar=False):
        return evaluate(y_true, self.predict(X), self.predict_proba(X), title=print_metrics,graficar=graficar)

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

class DecisionTree:
    def __init__(self, max_depth=None, min_samples_split=2, n_features=None,classes=None):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.n_features = n_features  # cantidad de features a usar en cada split
        self.tree = None
        self.classes = classes

    def fit(self, X, y,classes=None):
        if classes is None:
            self.classes_ = np.unique(y)
        else:
            self.classes_ = classes
        self.class_to_index = {cls: i for i, cls in enumerate(self.classes_)}
        self.n_classes = len(set(y))
        self.n_features = self.n_features or X.shape[1]
        self.tree = self._grow_tree(X, y)

    def _entropy(self, y):
        counts = np.bincount(y)
        probs = counts / len(y)
        return -np.sum([p * np.log2(p) for p in probs if p > 0])

    def _best_split(self, X, y, features):
        best_gain = -1
        split = None
        parent_entropy = self._entropy(y)

        for feat in features:
            thresholds = np.unique(X[:, feat])
            for threshold in thresholds:
                left = y[X[:, feat] <= threshold]
                right = y[X[:, feat] > threshold]
                if len(left) == 0 or len(right) == 0:
                    continue
                gain = parent_entropy - (
                    len(left) / len(y) * self._entropy(left)
                    + len(right) / len(y) * self._entropy(right)
                )
                if gain > best_gain:
                    best_gain = gain
                    split = (feat, threshold)
        return split

    def _grow_tree(self, X, y, depth=0):
        n_samples, n_features = X.shape
        if (depth >= self.max_depth or
            n_samples < self.min_samples_split or
            len(set(y)) == 1):
            return Counter(y).most_common(1)[0][0]

        feat_idxs = np.random.choice(n_features, self.n_features, replace=False)
        split = self._best_split(X, y, feat_idxs)
        if not split:
            return Counter(y).most_common(1)[0][0]

        feat, thresh = split
        left_idxs = X[:, feat] <= thresh
        right_idxs = X[:, feat] > thresh
        left = self._grow_tree(X[left_idxs], y[left_idxs], depth + 1)
        right = self._grow_tree(X[right_idxs], y[right_idxs], depth + 1)
        return (feat, thresh, left, right)

    def _predict_one(self, x, tree):
        if not isinstance(tree, tuple):
            return tree
        feat, thresh, left, right = tree
        if x[feat] <= thresh:
            return self._predict_one(x, left)
        else:
            return self._predict_one(x, right)

    def predict(self, X):
        return np.array([self._predict_one(x, self.tree) for x in X])
    
    def predict_proba(self, X):
        proba = np.zeros((X.shape[0], self.n_classes))
        for i, x in enumerate(X):
            class_label = self._predict_one(x, self.tree)
            class_index = self.class_to_index[class_label] 
            proba[i, class_index] += 1
        return proba


class RandomForestClassifier:
    def __init__(self, n_trees=10, max_depth=999, min_samples_split=2, max_features=None):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.trees = []

    def fit(self, X, y):
        X = X.to_numpy() if isinstance(X, pd.DataFrame) else X
        y = y.to_numpy() if isinstance(y, pd.Series) else y
        self.classes_ = np.unique(y)
        self.trees = []
        for _ in range(self.n_trees):
            idxs = np.random.choice(len(X), len(X), replace=True)
            X_sample, y_sample = X[idxs], y[idxs]
            tree = DecisionTree(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                n_features=self.max_features
            )
            tree.fit(X_sample, y_sample, classes=self.classes_)
            self.trees.append(tree)

    def predict(self, X):
        tree_preds = np.array([tree.predict(X) for tree in self.trees])
        return np.array([Counter(preds).most_common(1)[0][0] for preds in tree_preds.T])
    
    def predict_proba(self, X):
        tree_preds = np.array([tree.predict_proba(X) for tree in self.trees])
        return np.mean(tree_preds, axis=0)
    
    def evaluate(self, X, y_true, print_metrics="",graficar=False,rf=True):
        X = X.to_numpy() if isinstance(X, pd.DataFrame) else X
        y_true = y_true.to_numpy() if isinstance(y_true, pd.Series) else y_true
        return evaluate(y_true, self.predict(X), self.predict_proba(X), title=print_metrics,graficar=graficar,rf=rf)
