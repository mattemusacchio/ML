from IPython.display import display, Markdown
import pandas as pd
import numpy as np

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

def find_best_lambda(X_train, y_train, X_val, y_val,reweight=False):
    from .models import LogisticRegression
    """
    Busca el mejor valor de lambda para la regresión logística usando validación cruzada.
    """

    lambda_values = np.logspace(-4, 2, 10)

    # Inicializamos variables para encontrar el mejor lambda
    best_fscore = 0
    best_lambda = None

    # Probamos diferentes valores de lambda
    results = []

    for lambda_val in lambda_values:
        # Entrenamos el modelo con el lambda actual
        model = LogisticRegression(X_train, y_train, l2=lambda_val)
        model.fit_gradient_descent(reweight=reweight)
        
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
