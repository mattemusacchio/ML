import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def confusion_matrix(y_true, y_pred):
    """Calcula la matriz de confusión."""
    clases = np.unique(np.concatenate([y_true, y_pred]))
    n = len(clases)
    matriz = np.zeros((n, n), dtype=int)
    
    for i, real in enumerate(clases):
        for j, pred in enumerate(clases):
            matriz[i, j] = np.sum((y_true == real) & (y_pred == pred))
    
    return matriz

def accuracy_score(y_true, y_pred):
    """Calcula la precisión (accuracy) del modelo."""
    return np.mean(y_true == y_pred)

def precision_score(y_true, y_pred, average='macro'):
    """Calcula la precisión (precision) del modelo."""
    clases = np.unique(np.concatenate([y_true, y_pred]))
    precisiones = []

    for c in clases:
        tp = np.sum((y_true == c) & (y_pred == c))
        fp = np.sum((y_true != c) & (y_pred == c))
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        precisiones.append(precision)

    return np.mean(precisiones) if average == 'macro' else None

def recall_score(y_true, y_pred, average='macro'):
    """Calcula el recall del modelo."""
    clases = np.unique(np.concatenate([y_true, y_pred]))
    recalls = []

    for c in clases:
        tp = np.sum((y_true == c) & (y_pred == c))
        fn = np.sum((y_true == c) & (y_pred != c))
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        recalls.append(recall)

    return np.mean(recalls) if average == 'macro' else None

def f1_score(y_true, y_pred, average='macro'):
    """Calcula el F1-score del modelo."""
    clases = np.unique(np.concatenate([y_true, y_pred]))
    f1s = []

    for c in clases:
        tp = np.sum((y_true == c) & (y_pred == c))
        fp = np.sum((y_true != c) & (y_pred == c))
        fn = np.sum((y_true == c) & (y_pred != c))

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        f1s.append(f1)

    return np.mean(f1s) if average == 'macro' else None

def roc_auc_score(y_true, y_proba, average='macro'):
    """
    Calcula el AUC-ROC para clasificación multiclase usando One-vs-Rest.
    Devuelve listas de TPR y FPR por clase.
    """
    clases = np.unique(y_true)
    tpr_dict = {}
    fpr_dict = {}
    aucs = []

    for i, c in enumerate(clases):
        y_true_binary = (y_true == c).astype(int)
        y_score = y_proba[:, i]
        thresholds = np.sort(np.unique(y_score))[::-1]
        
        tpr_list = []
        fpr_list = []

        for threshold in thresholds:
            y_pred = (y_score >= threshold).astype(int)
            tp = np.sum((y_true_binary == 1) & (y_pred == 1))
            fn = np.sum((y_true_binary == 1) & (y_pred == 0))
            fp = np.sum((y_true_binary == 0) & (y_pred == 1))
            tn = np.sum((y_true_binary == 0) & (y_pred == 0))

            tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

            tpr_list.append(tpr)
            fpr_list.append(fpr)

        auc = np.trapz(tpr_list, fpr_list)
        aucs.append(auc)
        tpr_dict[c] = tpr_list
        fpr_dict[c] = fpr_list

    if average == 'macro':
        return tpr_dict, fpr_dict, np.mean(aucs)
    else:
        return tpr_dict, fpr_dict, aucs
    

def average_precision_score(y_true, y_proba, average='macro'):
    """
    Calcula el AUC-PR para clasificación multiclase usando One-vs-Rest.
    """
    clases = np.unique(y_true)
    precision_dict = {}
    recall_dict = {}
    aucs = []

    for i, c in enumerate(clases):
        y_true_binary = (y_true == c).astype(int)
        y_score = y_proba[:, i]
        thresholds = np.sort(np.unique(y_score))[::-1]

        precision_list = []
        recall_list = []

        for threshold in thresholds:
            y_pred = (y_score >= threshold).astype(int)

            tp = np.sum((y_true_binary == 1) & (y_pred == 1))
            fn = np.sum((y_true_binary == 1) & (y_pred == 0))
            fp = np.sum((y_true_binary == 0) & (y_pred == 1))

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

            precision_list.append(precision)
            recall_list.append(recall)

        auc = np.trapz(precision_list, recall_list)
        aucs.append(auc)
        precision_dict[c] = precision_list
        recall_dict[c] = recall_list

    if average == 'macro':
        return precision_dict, recall_dict, np.mean(aucs)
    else:
        return precision_dict, recall_dict, aucs


def average_precision_score_rf(y_true, y_proba, average='macro'):
    """
    Calcula el AUC-PR para clasificación multiclase usando One-vs-Rest.
    
    Parámetros:
    - y_true: array (n_samples,) con las etiquetas verdaderas (e.g. [1, 2, 3])
    - y_proba: array (n_samples, n_classes) con las probabilidades predichas por clase
    - average: 'macro' para devolver el promedio entre clases, o 'none' para devolver todos

    Retorna:
    - precision_dict: dict clase -> lista de precisión para distintos thresholds
    - recall_dict: dict clase -> lista de recall para distintos thresholds
    - auc_pr: float si average='macro', lista si average='none'
    """
    clases = np.unique(y_true)
    class_to_index = {label: idx for idx, label in enumerate(clases)}

    precision_dict = {}
    recall_dict = {}
    aucs = []

    for c in clases:
        idx = class_to_index[c]
        y_true_binary = (y_true == c).astype(int)
        y_score = y_proba[:, idx]

        # Ordenamos por score descendente
        sorted_indices = np.argsort(-y_score)
        y_true_sorted = y_true_binary[sorted_indices]
        y_score_sorted = y_score[sorted_indices]

        tp = 0
        fp = 0
        fn = np.sum(y_true_binary)
        precision_list = []
        recall_list = []

        for i in range(len(y_score_sorted)):
            if y_true_sorted[i] == 1:
                tp += 1
                fn -= 1
            else:
                fp += 1

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

            precision_list.append(precision)
            recall_list.append(recall)

        auc = np.trapz(precision_list, recall_list)
        aucs.append(auc)
        precision_dict[c] = precision_list
        recall_dict[c] = recall_list

    if average == 'macro':
        return precision_dict, recall_dict, np.mean(aucs)
    else:
        return precision_dict, recall_dict, aucs



def plot_roc_curve(y_true, y_proba):
    """
    Grafica la curva ROC para clasificación multiclase (One-vs-Rest).
    """
    tpr_dict, fpr_dict, macro_auc = roc_auc_score(y_true, y_proba)
    
    plt.figure(figsize=(6, 6))
    for clase in tpr_dict:
        plt.plot(fpr_dict[clase], tpr_dict[clase], label=f"Clase {clase}")
    
    plt.plot([0, 1], [0, 1], 'k--', linewidth=0.8)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"Curva ROC (AUC-ROC = {macro_auc:.2f})")
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_precision_recall_curve(y_true, y_proba):
    """
    Grafica la curva Precision-Recall para clasificación multiclase (One-vs-Rest).
    """
    precision_dict, recall_dict, macro_auc = average_precision_score_rf(y_true, y_proba)

    plt.figure(figsize=(6, 6))
    for clase in precision_dict:
        plt.plot(recall_dict[clase], precision_dict[clase], label=f"Clase {clase}")
    
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"Curva Precision-Recall (AUC-PR = {macro_auc:.2f})")
    plt.legend()
    plt.grid(True)
    plt.show()

