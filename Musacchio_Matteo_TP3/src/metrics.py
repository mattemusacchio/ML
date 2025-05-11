import cupy as cp  # cupy como cp para distinguirlo
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

def plot_confusion_matrix(cm, class_labels, title="Matriz de Confusión"):
    plt.figure(figsize=(12, 8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Greens)
    plt.colorbar()
    tick_marks = np.arange(len(class_labels))
    plt.xticks(tick_marks, class_labels, rotation=45)
    plt.yticks(tick_marks, class_labels)

    plt.ylabel('Etiqueta Real')
    plt.xlabel('Etiqueta Predicha')
    plt.tight_layout()
    plt.show()

def evaluate(model, X, y_true_oh, title="Dataset"):
    from .utils import pretty_print_df
    from .models import cross_entropy

    with torch.no_grad():
        output = model.forward(X)
        if isinstance(output, tuple):
            y_pred_proba = output[0][-1]
        else:
            y_pred_proba = output

        if isinstance(y_pred_proba, torch.Tensor):
            y_pred_proba = y_pred_proba.detach().cpu().numpy()
        elif isinstance(y_pred_proba, cp.ndarray):
            y_pred_proba = y_pred_proba.get()

        if isinstance(y_true_oh, torch.Tensor):
            y_true_oh = y_true_oh.detach().cpu().numpy()
        elif isinstance(y_true_oh, cp.ndarray):
            y_true_oh = y_true_oh.get()

        y_pred = np.argmax(y_pred_proba, axis=1)
        y_true = np.argmax(y_true_oh, axis=1)

    acc = np.mean(y_true == y_pred)
    loss = cross_entropy(cp.asarray(y_true_oh), cp.asarray(y_pred_proba))
    num_classes = y_true_oh.shape[1]

    cm = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1

    # En caso de que cm venga de CuPy
    if isinstance(cm, cp.ndarray):
        cm = cm.get()

    # Si las métricas vienen de cupy
    if isinstance(acc, cp.ndarray):
        acc = acc.get().item()
    if isinstance(loss, cp.ndarray):
        loss = loss.get().item()

    df_metrics = pd.DataFrame({
        "Métrica": ["Accuracy", "Cross-Entropy"],
        "Valor": [acc, loss]
    })

    pretty_print_df(df_metrics, title=f"Resultados en {title}", num_rows=None)

    class_labels = [str(i) for i in range(num_classes)]
    plot_confusion_matrix(cm, class_labels, title=f"Matriz de Confusión en {title}")

    return acc, loss, cm


def accuracy(y_true, y_pred):
    return np.mean(y_true == y_pred)


def confusion_matrix(y_true, y_pred, num_classes):
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    return cm