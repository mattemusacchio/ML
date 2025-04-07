import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def confusion_matrix(y_true, y_pred):
    """Calcula la matriz de confusión."""
    tp = np.sum((y_true == 1) & (y_pred == 1))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    return np.array([[tn, fp], [fn, tp]])

def accuracy_score(y_true, y_pred):
    """Calcula la precisión (accuracy) del modelo."""
    return np.mean(y_true == y_pred)

def precision_score(y_true, y_pred):
    """Calcula la precisión (precision) del modelo."""
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    return tp / (tp + fp) if (tp + fp) > 0 else 0.0

def recall_score(y_true, y_pred):
    """Calcula el recall del modelo."""
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    return tp / (tp + fn) if (tp + fn) > 0 else 0.0

def f1_score(y_true, y_pred):
    """Calcula el F1-score del modelo."""
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    return 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

def roc_auc_score(y_true, y_proba):
    """Calcula el AUC-ROC usando integración numérica."""
    thresholds = np.sort(y_proba)[::-1]
    tpr_list = []
    fpr_list = []
    
    for threshold in thresholds:
        y_pred = (y_proba >= threshold).astype(int)
        tp = np.sum((y_true == 1) & (y_pred == 1))
        tn = np.sum((y_true == 0) & (y_pred == 0))
        fp = np.sum((y_true == 0) & (y_pred == 1))
        fn = np.sum((y_true == 1) & (y_pred == 0))
        
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        
        tpr_list.append(tpr)
        fpr_list.append(fpr)

    return tpr_list, fpr_list 

def average_precision_score(y_true, y_proba):
    """Calcula el AUC-PR usando integración numérica."""
    thresholds = np.sort(y_proba)[::-1]
    precision_list = []
    recall_list = []
    
    for threshold in thresholds:
        y_pred = (y_proba >= threshold).astype(int)
        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        precision_list.append(precision)
        recall_list.append(recall)

    return precision_list, recall_list

def plot_roc_curve(y_true, y_proba):
    """Grafica la curva ROC."""

    tpr_list, fpr_list = roc_auc_score(y_true, y_proba)

    plt.figure(figsize=(6, 6))
    plt.plot(fpr_list, tpr_list, label=f"AUC-ROC = {np.trapz(tpr_list,fpr_list):.2f}", color="blue")
    plt.plot([0, 1], [0, 1], 'k--', linewidth=0.8)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Curva ROC")
    plt.legend()
    plt.show()

def plot_precision_recall_curve(y_true, y_proba):
    """Grafica la curva Precision-Recall."""

    precision_list,recall_list = average_precision_score(y_true, y_proba)

    plt.figure(figsize=(6, 6))
    plt.plot(recall_list, precision_list, label=f"AUC-PR = {np.trapz(precision_list,recall_list):.2f}", color="red")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Curva Precision-Recall")
    plt.legend()
    plt.show()