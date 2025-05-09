import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

def plot_confusion_matrix(cm, class_labels, title="Matriz de Confusión"):
    plt.figure(figsize=(6, 5))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(class_labels))
    plt.xticks(tick_marks, class_labels, rotation=45)
    plt.yticks(tick_marks, class_labels)

    # Mostrar los valores en las celdas
    # thresh = cm.max() / 2.
    # for i in range(len(class_labels)):
    #     for j in range(len(class_labels)):
    #         plt.text(j, i, format(cm[i, j], 'd'),
    #                  ha="center", va="center",
    #                  color="white" if cm[i, j] > thresh else "black")

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
            y_pred_proba = output[0][-1]  # modelo devuelve (all_outputs, ...)
        else:
            y_pred_proba = output

        # convertir a numpy si es tensor
        if isinstance(y_pred_proba, torch.Tensor):
            y_pred_proba = y_pred_proba.detach().numpy()
        if isinstance(y_true_oh, torch.Tensor):
            y_true_oh = y_true_oh.detach().numpy()

        y_pred = np.argmax(y_pred_proba, axis=1)
        y_true = np.argmax(y_true_oh, axis=1)
    
    # ... seguir con métricas y visualización


    acc = np.mean(y_true == y_pred)
    loss = cross_entropy(y_true_oh, y_pred_proba)
    num_classes = y_true_oh.shape[1]
    cm = np.zeros((y_true_oh.shape[1], y_true_oh.shape[1]), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1

    # Armar el dataframe con métricas
    df_metrics = pd.DataFrame({
        "Métrica": ["Accuracy", "Cross-Entropy"],
        "Valor": [acc, loss]
    })

    # Pretty print de las métricas
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