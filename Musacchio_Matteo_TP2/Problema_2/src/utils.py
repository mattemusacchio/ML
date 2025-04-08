import numpy as np
import pandas as pd
from IPython.display import display, Markdown

def compute_entropy(counts):
    """Calcula la entropía H(X) a partir de cuentas"""
    probs = counts / counts.sum()
    probs = probs[probs > 0]  # Evita log(0)
    return -np.sum(probs * np.log2(probs))

def conditional_entropy(y, x, n_bins=10):
    """Calcula la entropía condicional H(Y|X) discretizando X"""
    df = pd.DataFrame({'x': x, 'y': y})
    # Discretizamos X en quantiles
    df['x_bin'] = pd.qcut(df['x'], q=n_bins, duplicates='drop')
    H_y_given_x = 0
    total = len(df)
    
    for x_val in df['x_bin'].unique():
        subset = df[df['x_bin'] == x_val]['y']
        weight = len(subset) / total
        H_subset = compute_entropy(subset.value_counts())
        H_y_given_x += weight * H_subset
    return H_y_given_x

def mutual_information_feature(y, x, n_bins=10):
    """Calcula I(X; Y) = H(Y) - H(Y|X)"""
    H_y = compute_entropy(pd.Series(y).value_counts())
    H_y_given_x = conditional_entropy(y, x, n_bins)
    return H_y - H_y_given_x

def mutual_information_all(X, y, n_bins=10):
    """Calcula I(Xi ; y) para cada feature numérico Xi"""
    mi_scores = {}
    for col in X.columns:
        if pd.api.types.is_numeric_dtype(X[col]):
            try:
                mi = mutual_information_feature(y, X[col], n_bins)
                mi_scores[col] = mi
            except Exception as e:
                print(f"Error en feature {col}: {e}")
    return pd.Series(mi_scores).sort_values(ascending=False)

def pretty_print_df(df, num_rows=15, title=None, index=False):
    """
    Imprime un DataFrame de pandas en formato Markdown para una mejor visualización.
    
    Args:
        df (pandas.DataFrame): El DataFrame a imprimir.
        num_rows (int, opcional): Número de filas a mostrar. Por defecto es 5.
        title (str, opcional): Título para la tabla. Por defecto es None.
    
    Returns:
        None: La función muestra el DataFrame directamente.
    """
    if df is None or len(df) == 0:
        display(Markdown("*DataFrame vacío*"))
        return
    
    # Limitar el número de filas si es necesario
    if num_rows is not None and len(df) > num_rows:
        df_display = df.head(num_rows)
    else:
        df_display = df
    
    # Crear el markdown
    markdown_text = ""
    
    # Agregar título si existe
    if title:
        markdown_text += f"### {title}\n\n"
    
    # Convertir DataFrame a markdown
    markdown_text += df_display.to_markdown(index=index)
    
    # Agregar información sobre filas totales si se limitó
    if num_rows is not None and len(df) > num_rows:
        markdown_text += f"\n\n*Mostrando {num_rows} de {len(df)} filas*"
    
    # Mostrar el markdown
    display(Markdown(markdown_text))


def find_best_lambda_m(X_train, y_train, X_val, y_val):
    from .models import MulticlassLogisticRegression
    """
    Busca el mejor valor de lambda para la regresión logística usando validación cruzada.
    """

    lambda_values = np.logspace(-14, -1, 10)

    # Inicializamos variables para encontrar el mejor lambda
    best_fscore = 0
    best_lambda = None

    # Probamos diferentes valores de lambda
    results = []

    for lambda_val in lambda_values:
        # Entrenamos el modelo con el lambda actual
        model = MulticlassLogisticRegression(X_train, y_train, l2=lambda_val)
        model.fit_gradient_descent()
        
        # Evaluamos el modelo
        metrics = model.evaluate(X_val, y_val)
        fscore = metrics['F1-Score']
        
        # Guardamos los resultados
        results.append({
            'Lambda': lambda_val,
            'F1-Score': fscore
        })
        
        # Actualizamos el mejor lambda si encontramos un mejor F1-Score
        if fscore > best_fscore:
            best_fscore = fscore
            best_lambda = lambda_val

    print(f"\nMejor lambda encontrado: {best_lambda} (F1-Score: {best_fscore:.4f})")
    return best_lambda
