import pandas as pd
from IPython.display import display, Markdown
from metrics import MSE, MAE, RMSE, R2
from models import LinearRegression
import numpy as np
import matplotlib.pyplot as plt

def pretty_print_df(df, num_rows=15, title=None,index=False):
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

   
def trainPredictAndImport(train: pd.DataFrame, validation: pd.DataFrame, feature: str ='all', method: str ='both',l1=0.0,l2=0.0):
    """Esta función entrena un modelo de regresión lineal y evalúa su desempeño en un conjunto de validación.

    Args:
        train (pd.DataFrame): DataFrame con los datos de entrenamiento.
        validation (pd.DataFrame): DataFrame con los datos de validación.
        feature (str, optional): Nombre de la columna a utilizar como feature. Si se pasa 'all', se utilizan todas las columnas excepto 'price'. Defaults to 'all'.
        method (str, optional): Método de entrenamiento a utilizar. Puede ser 'gradient', 'pseudo' o 'both'. Defaults to 'both'.
    """
    def evaluateGradientDescent(model: LinearRegression, X: pd.DataFrame, y: pd.Series):
        print('Resultados para Descenso por Gradiente')
        model.fit_gradient_descent()
        model.predict(train_df)
        model.print_coefficients()
        print('Métricas en datos de entrenamiento:')
        model.compute_loss(metrics='all', X=train_df, Y=train_y_log)
        print('Métricas en datos de validación:')
        model.compute_loss(metrics='all', X=val_df, Y=val_y_log)
    
    def evaluatePseudoInverse(model: LinearRegression, X: pd.DataFrame, y: pd.Series):
        print('Results for Pseudo Inverse')
        model.fit_pseudo_inverse()
        model.predict(train_df)
        model.print_coefficients()
        print('Métricas en datos de entrenamiento:')
        model.compute_loss(metrics='all', X=train_df, Y=train_y_log)
        print('Métricas en datos de validación:')
        model.compute_loss(metrics='all', X=val_df, Y=val_y_log)

    def evaluateNormalEquation(model: LinearRegression, X: pd.DataFrame, y: pd.Series):
        print('Results for Normal Equation')
        model.fit_normal_equation()
        model.predict(train_df)
        model.print_coefficients()
        print('Métricas en datos de entrenamiento:')
        model.compute_loss(metrics='all', X=train_df, Y=train_y_log)
        print('Métricas en datos de validación:')
        model.compute_loss(metrics='all', X=val_df, Y=val_y_log)

    train_y = train['price']
    train_y_log = np.log1p(train_y)
    val_y = validation['price']
    val_y_log = np.log1p(val_y)

    if feature == 'all':
        train_df = train.drop(columns=['price'])
        val_df = validation.drop(columns=['price'])
    elif isinstance(feature, list):
        train_df = train[feature]
        val_df = validation[feature]
    else:
        train_df = train[[feature]]
        val_df = validation[[feature]] 
    
    model = LinearRegression(train_df, train_y_log)
    if method == 'gradient':
        evaluateGradientDescent(model, val_df, val_y)
    elif method == 'pseudo':
        evaluatePseudoInverse(model, val_df, val_y)
    if method == 'both':
        # Generate two new models to avoid overwriting the previous one
        evaluateGradientDescent(LinearRegression(train_df, train_y_log), val_df, val_y)
        print('========================================================')
        evaluatePseudoInverse(LinearRegression(train_df, train_y_log), val_df, val_y)
    if method == 'l2':
        evaluateNormalEquation(LinearRegression(train_df, train_y_log, l2=l2), val_df, val_y)
    if method == 'l1':
        evaluateGradientDescent(LinearRegression(train_df, train_y_log, l1=l1), val_df, val_y)

def feature_engineering(df):
    from preprocessing import one_hot_encoder
    """
    Realiza ingeniería de características en el DataFrame de propiedades.
    
    Esta función:
    1. Aplica one-hot encoding a area_units
    2. Crea clusters basados en area_units_m2
    3. Calcula el centro de cada cluster (lat, lon)
    4. Calcula la distancia de cada propiedad al centro de su cluster
    
    Args:
        df (pd.DataFrame): DataFrame con los datos de propiedades
        
    Returns:
        pd.DataFrame: DataFrame con las nuevas características
    """
    # Usar one hot encoding para dividir area_units entre sqft y m2 y llamarlo cluster
    df, encoder_dict = one_hot_encoder(df, ['area_units'])
    
    # Crear nuevas características derivadas
    df['cluster'] = df['area_units_m2']
    
    # Eliminar las columnas area_units_m2 y area_units_sqft
    df.drop(columns=['area_units_m2', 'area_units_sqft'], inplace=True) 
    
    # Calcular el centro de cada cluster
    clusters = df['cluster'].unique()
    cluster_center = []
    
    for cluster in clusters:
        cluster_data = df[df['cluster'] == cluster]
        center_lat = cluster_data['lat'].mean()
        center_lon = cluster_data['lon'].mean()
        cluster_center.append([cluster, center_lat, center_lon])
    
    cluster_center = pd.DataFrame(cluster_center, columns=['cluster', 'center_lat', 'center_lon'])
    
    # Unir el centro del cluster con el DataFrame original
    df = df.merge(cluster_center, on='cluster', how='left')
    
    # Calcular la distancia desde el centro del cluster
    df['distance_from_center'] = np.sqrt((df['lat'] - df['center_lat'])**2 + (df['lon'] - df['center_lon'])**2)

    # Calcular área por habitación (area per room)
    df['area_per_room'] = df['area'] / df['rooms']

    # Aplicar transformación logarítmica a las columnas 'area', 'rooms' y 'age'
    for col in ['area', 'rooms', 'age']:
        if col in df.columns and (df[col] > 0).all():
            skewness_before = df[col].skew()
            df[f'log_{col}'] = np.log(df[col])
            skewness_after = df[f'log_{col}'].skew()
            # print(f"Asimetría de {col}: antes={skewness_before:.4f}, después={skewness_after:.4f}")
    
    return df

