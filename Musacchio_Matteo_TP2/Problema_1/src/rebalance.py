import numpy as np
import pandas as pd

def undersampling(X, y, random_state=42):
    """
    Undersample the majority class in a dataset.

    Parameters:
    - X: Features of the dataset.
    - y: Labels of the dataset.
    - random_state: Random state for reproducibility.

    Returns:
    - X_undersampled: Undersampled features.
    - y_undersampled: Undersampled labels.
    """
    
    # Establecemos la semilla para la reproducibilidad
    np.random.seed(random_state)
    # Contamos la cantidad de muestras por clase
    class_counts = y.value_counts()
    majority_class = class_counts.idxmax()
    minority_class = class_counts.idxmin()
    majority_count = class_counts.max()
    minority_count = class_counts.min()
    samples_to_remove = majority_count - minority_count
    majority_indices = y[y == majority_class].index.tolist()
    random_indices = np.random.choice(majority_indices, samples_to_remove, replace=False)
    X_under = X.drop(index=random_indices)
    y_under = y.drop(index=random_indices)

    return X_under, y_under

def smote_oversample(X, y):
    """
    Genera muestras sintéticas mediante SMOTE para la clase minoritaria.
    
    Parámetros:
        X (pd.DataFrame): Conjunto de características.
        y (pd.Series): Etiquetas.
        target_class (int/float): Clase minoritaria.
        n_samples (int): Número de muestras sintéticas a generar.
        
    Retorna:
        (X_new, y_new): Nuevas muestras sintéticas.
    """
    np.random.seed(412)

    class_counts = y.value_counts()
    target_class = class_counts.idxmin()
    majority_count = class_counts.max()
    minority_count = class_counts.min()

    # Cuántas muestras sintéticas necesitamos
    n_samples = majority_count - minority_count

    X_min = X[y == target_class].values
    synthetic_samples = []

    for _ in range(n_samples):
        i = np.random.randint(0, len(X_min))
        j = np.random.randint(0, len(X_min))
        while j == i:
            j = np.random.randint(0, len(X_min))

        x_i = X_min[i]
        x_j = X_min[j]
        alpha = np.random.rand()
        new_sample = x_i + alpha * (x_j - x_i)
        synthetic_samples.append(new_sample)

    X_synth = pd.DataFrame(synthetic_samples, columns=X.columns)
    y_synth = pd.Series([target_class] * n_samples, name='Diagnosis')
    
    return X_synth, y_synth

def oversampling(X, y):
    np.random.seed(43)
    class_counts = y.value_counts()
    # Encontramos la clase mayoritaria y minoritaria
    majority_class = class_counts.idxmax()
    minority_class = class_counts.idxmin()
    majority_count = class_counts.max()
    minority_count = class_counts.min()
    # Calculamos la cantidad de muestras a duplicar de la clase minoritaria
    samples_to_duplicate = majority_count - minority_count
    # Obtenemos los índices de las muestras de la clase minoritaria
    minority_indices = y[y == minority_class].index.tolist()
    # Seleccionamos aleatoriamente los índices a duplicar
    random_indices = np.random.choice(minority_indices, samples_to_duplicate, replace=True)
    # Duplicamos las muestras seleccionadas del conjunto de entrenamiento
    X_dupe = pd.concat([X, X.loc[random_indices]])
    y_dupe = pd.concat([y, y.loc[random_indices]])

    return X_dupe, y_dupe
