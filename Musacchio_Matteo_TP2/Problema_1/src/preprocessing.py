import numpy as np
import pandas as pd

def one_hot_encoding(df, categorical_columns=None):
    """
    Realiza one-hot encoding en columnas categóricas, tratando '???' y NaN como una categoría separada (_Nan).
    
    Args:
        df (pd.DataFrame): DataFrame con columnas categóricas
        categorical_columns (list, opcional): Lista de columnas categóricas a codificar.
                                              Si es None, se detectan automáticamente.
    
    Returns:
        pd.DataFrame: DataFrame con las columnas codificadas
    """
    df_copy = df.copy()

    if categorical_columns is None:
        categorical_columns = df_copy.select_dtypes(include=['object', 'category']).columns.tolist()

    for col in categorical_columns:
        if col in df_copy.columns:
            # Unificar '???' y NaN como np.nan
            df_copy[col] = df_copy[col].replace('???', np.nan)

            # Obtener valores únicos excluyendo NaN
            unique_values = df_copy[col].dropna().unique()

            for value in unique_values:
                new_col_name = f"{col}_{value}"
                df_copy[new_col_name] = (df_copy[col] == value).astype(int)

            # Crear columna para los NaN (incluye antes '???')
            nan_col_name = f"{col}_Nan"
            df_copy[nan_col_name] = np.where(df_copy[col].isna(), np.nan, 0)

            # Eliminar la columna original
            df_copy.drop(columns=[col], inplace=True)

    return df_copy


    
def handle_missing_values(df, n_neighbors=7):
    """
    Maneja valores faltantes en un DataFrame usando KNN.
    
    Args:
        df (pd.DataFrame): DataFrame con valores faltantes
        n_neighbors (int): Número de vecinos a considerar
        
    Returns:
        pd.DataFrame: DataFrame con valores faltantes imputados
    """
    
    # Crear una copia del DataFrame para evitar modificar el original
    df_copy = df.copy()
    
    # Identificar columnas numéricas
    numeric_cols = df_copy.select_dtypes(include=[np.number]).columns
    
    # Convertir DataFrame a numpy para cálculos eficientes
    data = df_copy[numeric_cols].to_numpy()
    mask = np.isnan(data)
    
    # Calcular imputación basada en vecinos más cercanos
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            if mask[i, j]:
                # Calcular distancias a otras filas sin valores faltantes en la columna j
                distances = []
                values = []
                for k in range(data.shape[0]):
                    if not mask[k, j]:
                        dist = np.nansum((data[i, :] - data[k, :]) ** 2)  # Distancia euclidiana
                        distances.append((dist, data[k, j]))
                
                # Seleccionar los n_neighbors más cercanos y calcular la media
                distances.sort()
                neighbors = [val for _, val in distances[:n_neighbors]]
                data[i, j] = np.mean(neighbors) if neighbors else np.nan
    
    # Restaurar los valores imputados al DataFrame
    df_copy[numeric_cols] = data
    
    return df_copy

def handle_outliers(df, method='nan', threshold=1.5):
    """
    Detecta y maneja outliers en un DataFrame.
    
    Args:
        df (pd.DataFrame): DataFrame con posibles outliers
        method (str): Método para reemplazar outliers ('mean', 'median', 'clip')
        threshold (float): Factor para el cálculo de los límites (default: 1.5)
        
    Returns:
        pd.DataFrame: DataFrame con outliers manejados
    """
    # Crear una copia para no modificar el original
    df_copy = df.copy()
    
    # Identificar columnas numéricas
    numeric_columns = df_copy.select_dtypes(include=['int64', 'float64']).columns
    
    # Para cada columna numérica
    for col in numeric_columns:
        # Calcular Q1, Q3 e IQR
        Q1 = df_copy[col].quantile(0.25)
        Q3 = df_copy[col].quantile(0.75)
        IQR = Q3 - Q1
        
        # # Calcular límites
        lower_bound = Q1 - threshold * IQR
        upper_bound = Q3 + threshold * IQR
        
        # Identificar outliers
        outliers = (df_copy[col] < lower_bound) | (df_copy[col] > upper_bound)
        
        if outliers.any():
            # Contar outliers
            n_outliers = outliers.sum()
            outlier_percentage = (n_outliers / len(df_copy)) * 100
            
            # print(f"Columna '{col}': {n_outliers} outliers ({outlier_percentage:.2f}%)")
            
            # Reemplazar outliers según el método especificado
            if method == 'mean':
                # Reemplazar con la media de los valores no-outliers
                mean_value = df_copy.loc[~outliers, col].mean()
                df_copy.loc[outliers, col] = mean_value
            elif method == 'median':
                # Reemplazar con la mediana de los valores no-outliers
                median_value = df_copy.loc[~outliers, col].median()
                df_copy.loc[outliers, col] = median_value
            elif method == 'clip':
                # Recortar los valores a los límites
                df_copy.loc[df_copy[col] < lower_bound, col] = lower_bound
                df_copy.loc[df_copy[col] > upper_bound, col] = upper_bound
            elif method == 'delete':
                # Eliminar filas con outliers
                df_copy = df_copy[~outliers]
                print(f"Se eliminaron {n_outliers} filas con outliers en la columna '{col}'")
            elif method == 'nan':
                # Reemplazar outliers con NaN
                df_copy.loc[outliers, col] = np.nan
    
    return df_copy


def min_max_normalize(train_df=None, val_df=None, columns=None,params={}):
    """
    Normaliza las columnas numéricas seleccionadas usando normalización Min-Max,
    que escala los valores al rango [0,1]. Primero normaliza el conjunto de entrenamiento
    y luego aplica los mismos parámetros al conjunto de validación.
    
    Parámetros:
    - train_df: pd.DataFrame, el DataFrame de entrenamiento
    - val_df: pd.DataFrame, el DataFrame de validación
    - columns: list, lista de columnas a normalizar
    
    Retorna:
    - train_df: pd.DataFrame de entrenamiento normalizado
    - val_df: pd.DataFrame de validación normalizado
    - params: dict con los parámetros (min, max) por columna para poder revertir
    """
    # Primero normalizar el conjunto de entrenamiento y guardar parámetros
    if params == {}:
        for col in columns:
            min_val = train_df[col].min()
            max_val = train_df[col].max()
            train_df[col] = (train_df[col] - min_val) / (max_val - min_val)
            params[col] = (min_val, max_val)
    
    # Usar los parámetros del conjunto de entrenamiento para normalizar validación
    if val_df is not None:
        for col in params.keys():
                min_val, max_val = params[col]
                val_df[col] = (val_df[col] - min_val) / (max_val - min_val)
        
    return train_df, val_df, params

def preprocess_file(df, n_neighbors=7, outlier_method='mean', threshold=1.5,target_column='Diagnosis',params={}):
    """
    Preprocesa un DataFrame aplicando one-hot encoding, imputación de valores faltantes y manejo de outliers.
    
    Args:
        df (pd.DataFrame): DataFrame a preprocesar
        categorical_columns (list, opcional): Lista de columnas categóricas a codificar.
                                              Si es None, se detectan automáticamente.
        n_neighbors (int): Número de vecinos a considerar para la imputación KNN
        outlier_method (str): Método para manejar outliers ('mean', 'median', 'clip', 'delete')
        threshold (float): Factor para el cálculo de los límites de outliers
        
    Returns:
        pd.DataFrame: DataFrame preprocesado
    """
    X = df.drop(target_column, axis=1)
    y = df[target_column]
    X = handle_outliers(X)
    X = handle_missing_values(X)
    categorical_columns = X.select_dtypes(include=['object', 'category']).columns.tolist()
    X = one_hot_encoding(X, categorical_columns)
    if 'CellType_Mesnchymal' in X.columns and 'GeneticMutation_Absnt'in X.columns and 'GeneticMutation_Nan'in X.columns and 'CellType_Nan'in X.columns:
        X['CellType_Epthlial'] = X['CellType_Nan'] + X['CellType_Epthlial']
        X = X.drop(columns=['CellType_Nan', 'CellType_Mesnchymal','GeneticMutation_Absnt','GeneticMutation_Nan'])
        X['CellType_Epthlial'] = (X['CellType_Epthlial'] > 0.5).astype(int)
    # X = X.values.astype(np.float64)
    # y = y.values.astype(np.float64)
    df = pd.concat([X, y], axis=1)
    # if params == {}:
    #     df, _, params = min_max_normalize(train_df=df, columns=X.columns)
    #     return df, params
    # else:
    #     _, df, _ = min_max_normalize(val_df=df, params=params)
    return df


