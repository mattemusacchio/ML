import pandas as pd
from IPython.display import display, Markdown
from metrics import MSE, MAE, RMSE, R2
from models import LinearRegression
import numpy as np
import matplotlib.pyplot as plt

def pretty_print_df(df, num_rows=10, title=None,index=False):
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

def inverse_normalize(df, normalization_parameters):
    for col in normalization_parameters:
        min_val, max_val = normalization_parameters[col]
        df[col] = df[col] * (max_val - min_val) + min_val
    return df

def print_metrics(y_true, y_pred, model_name=""):
    """
    Calcula y muestra todas las métricas en un DataFrame formateado.
    
    Parámetros:
    - y_true: np.array, valores reales
    - y_pred: np.array, valores predichos
    - model_name: str, nombre del modelo a evaluar
    """
    metrics = {
        'Métrica': ['MSE', 'MAE', 'RMSE', 'R2'],
        'Valor': [
            MSE(y_true, y_pred),
            MAE(y_true, y_pred),
            RMSE(y_true, y_pred),
            R2(y_true, y_pred)
        ]
    }
    df_metrics = pd.DataFrame(metrics)
    title = f"Métricas del modelo {model_name}" if model_name else "Métricas del modelo"
    pretty_print_df(df_metrics, title=title)

   
def trainPredictAndImport(train: pd.DataFrame, validation: pd.DataFrame, feature: str ='all', method: str ='both'):
    """Esta función entrena un modelo de regresión lineal y evalúa su desempeño en un conjunto de validación.

    Args:
        train (pd.DataFrame): DataFrame con los datos de entrenamiento.
        validation (pd.DataFrame): DataFrame con los datos de validación.
        feature (str, optional): Nombre de la columna a utilizar como feature. Si se pasa 'all', se utilizan todas las columnas excepto 'price'. Defaults to 'all'.
        method (str, optional): Método de entrenamiento a utilizar. Puede ser 'gradient', 'pseudo' o 'both'. Defaults to 'both'.
    """
    def evaluateGradientDescent(model: LinearRegression, X: pd.DataFrame, y: pd.Series):
        print('Results for Gradient Descent')
        model.fit_gradient_descent()
        predictions = model.predict(val_df)
        model.print_coefficients()
        print('Switching to validation data')
        model.compute_loss(metrics='all', X=val_df, Y=val_y)
        negative_predictions = predictions[predictions < 0]
    
    def evaluatePseudoInverse(model: LinearRegression, X: pd.DataFrame, y: pd.Series):
        print('Results for Pseudo Inverse')
        model.fit_pseudo_inverse()
        model.predict(val_df)
        model.print_coefficients()
        print('Switching to validation data')
        model.compute_loss(metrics='all', X=val_df, Y=val_y)

    train_y = train['price']
    val_y = validation['price']

    if feature == 'all':
        train_df = train.drop(columns=['price'])
        val_df = validation.drop(columns=['price'])
    elif isinstance(feature, list):
        train_df = train[feature]
        val_df = validation[feature]
    else:
        train_df = train[[feature]]
        val_df = validation[[feature]] 
    
    model = LinearRegression(train_df, train_y)
    if method == 'gradient':
        evaluateGradientDescent(model, val_df, val_y)
    elif method == 'pseudo':
        evaluatePseudoInverse(model, val_df, val_y)
    if method == 'both':
        # Generate two new models to avoid overwriting the previous one
        evaluateGradientDescent(LinearRegression(train_df, train_y), val_df, val_y)
        print('========================================================')
        evaluatePseudoInverse(LinearRegression(train_df, train_y), val_df, val_y)
        





