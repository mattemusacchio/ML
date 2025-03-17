def handle_missing_values(df):
    """
    Imputa valores faltantes en un DataFrame de manera general:
    - Para variables numéricas, usa la mediana.
    - Para variables categóricas, usa el valor más frecuente.
    """
    for column in df.columns:
        if df[column].isnull().sum() > 0:  # Solo procesar columnas con valores faltantes
            if df[column].dtype == 'object':  # Variables categóricas
                df[column] = df[column].fillna(df[column].mode()[0])
            else:  # Variables numéricas
                df[column] = df[column].fillna(df[column].median())
    return df

def normalize(df, columns):
    """
    Normaliza las columnas numéricas seleccionadas usando media y desviación estándar.
    Retorna el DataFrame normalizado y los parámetros de normalización para deshacer el proceso si es necesario.
    """
    params = {}
    for col in columns:
        mean = df[col].mean()
        std = df[col].std()
        df[col] = (df[col] - mean) / std
        params[col] = (mean, std)
    return df, params

def min_max_normalize(df, columns):
    """
    Normaliza las columnas numéricas seleccionadas usando normalización Min-Max,
    que escala los valores al rango [0,1].
    
    Parámetros:
    - df: pd.DataFrame, el DataFrame con los datos
    - columns: list, lista de columnas a normalizar
    
    Retorna:
    - df: pd.DataFrame normalizado
    - params: dict con los parámetros (min, max) por columna para poder revertir
    """
    params = {}
    for col in columns:
        min_val = df[col].min()
        max_val = df[col].max()
        df[col] = (df[col] - min_val) / (max_val - min_val)
        params[col] = (min_val, max_val)
    return df, params


def one_hot_encoder(df, columns):
    """
    Aplica codificación one-hot a las columnas categóricas especificadas.

    Parámetros:
    - df: pd.DataFrame, el DataFrame con los datos.
    - columns: list, lista de columnas categóricas a codificar.

    Retorna:
    - df_encoded: pd.DataFrame, DataFrame con las columnas codificadas.
    - encoder_dict: dict, diccionario con los nombres originales de las columnas y sus categorías.
    """
    df_encoded = df.copy()
    encoder_dict = {}

    for col in columns:
        # Obtener las categorías únicas de la columna
        unique_values = df_encoded[col].unique()
        encoder_dict[col] = unique_values

        # Crear columnas one-hot
        for val in unique_values:
            df_encoded[f"{col}_{val}"] = (df_encoded[col] == val).astype(int)

        # Eliminar la columna original
        df_encoded.drop(columns=[col], inplace=True)

    return df_encoded, encoder_dict


