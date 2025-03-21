import numpy as np
import pandas as pd
from typing import Tuple, Optional

def one_hot_encoder(data: pd.DataFrame, columns: list) -> pd.DataFrame:
    """Esta función aplica la codificación one-hot a las columnas especificadas.

    Args:
        data (pd.DataFrame): El DataFrame con los datos.
        columns (list): Las columnas a codificar.

    Returns:
        pd.DataFrame: El DataFrame con las columnas codificadas.
    """
    if type(columns) == str:
        columns = [columns]
    return pd.get_dummies(data, columns=columns)

def gaussian_normalization(data: pd.DataFrame, columns: list = None) -> Tuple[pd.DataFrame, dict]:
    """Esta función normaliza las columnas de un DataFrame según el método Gaussiano.

    Args:
        data (pd.DataFrame): El DataFrame con los datos.
        columns (list, optional): Las columnas a normalizar. Si es None, se normalizan todas las columnas numéricas. Defaults to None.

    Returns:
        pd.DataFrame: El DataFrame con las columnas normalizadas.
        dict: Un diccionario con los valores para desnornalizar los datos del precio.
    """
    if columns is None:
        columns = data.select_dtypes(include=['number']).columns
    stats = {"method": "gaussian"}
    
    for column in columns:
        mean = data[column].mean()
        std = data[column].std()
        data[column] = (data[column] - mean) / std
        stats[column] = {"mean": mean, "std": std}
    
    return data, stats

def min_max_normalization(data: pd.DataFrame, columns: list = None) -> Tuple[pd.DataFrame, dict]:
    """Esta función normaliza las columnas de un DataFrame según el método Min-Max.
    
    Args:
        data (pd.DataFrame): El DataFrame con los datos.
        columns (list, optional): Las columnas a normalizar. Si es None, se normalizan todas las columnas numéricas. Defaults to None.
        
    Returns:
        pd.DataFrame: El DataFrame con las columnas normalizadas.
        dict: Un diccionario con los valores para desnornalizar los datos del precio.
    """
    if columns is None:
        columns = data.select_dtypes(include=['number']).columns
    stats = {"method": "minmax"}
    
    for column in columns:
        min_val = data[column].min()
        max_val = data[column].max()
        data[column] = (data[column] - min_val) / (max_val - min_val)
        stats[column] = {"min": min_val, "max": max_val}
    
    return data, stats

    
def sqft2m2(data: float, reverse: bool = False) -> float:
    """Esta función convierte pies cuadrados a metros cuadrados las columnas especificadas.

    Args:
        data (float): El valor en pies cuadrados.
        reverse (bool): Si es False, convieerte de sqft a metros cuadrados. Defaults to False.

    Returns:
        float: El valor en metros cuadrados o pies cuadrados.
    """
    return data * 0.092903 if not reverse else data / 0.092903

def normalize_df(df: pd.DataFrame, method: str, columns: list = None, area_units: str = 'm2', normalize = True, normalize_dict: dict = None) -> Tuple[pd.DataFrame, dict]:
    """Esta función normaliza las columnas de un DataFrame según el método especificado.

    Args:
        df (pd.DataFrame): El DataFrame con los datos.
        method (str): El método de normalización a aplicar. Puede ser 'minmax' o 'gaussian'.
        columns (list, optional): Las columnas a normalizar. Si es None, se normalizan todas las columnas numéricas. Defaults to None.

    Returns:
        pd.DataFrame: El DataFrame con las columnas normalizadas.
        dict: Un diccionario con los valores para desnornalizar los datos del precio.
    """
    
    normalized_df = df.copy()
    stats = None
    
    if 'area_units' in df.columns:
        normalized_df['area'] = df.apply(lambda row: sqft2m2(row['area'], reverse=False) if row['area_units'] == 'sqft' else row['area'], axis=1)
        normalized_df = normalized_df.drop('area_units', axis=1)
    
def normalize_df(df: pd.DataFrame, method: str = 'gaussian', columns: list = None, area_units: str = 'm2', normalize=True, normalize_dict: dict = None) -> Tuple[pd.DataFrame, Optional[dict]]:
    """Esta función normaliza las columnas de un DataFrame según el método especificado.
    Además, convierte las áreas de pies cuadrados a metros cuadrados.
    
    Args:
        df (pd.DataFrame): El DataFrame con los datos.
        method (str): El método de normalización a aplicar. Puede ser 'minmax' o 'gaussian'.
        columns (list, optional): Las columnas a normalizar. Si es None, se normalizan todas las columnas numéricas.
        area_units (str, optional): Unidad del área. Default es 'm2'.
        normalize (bool, optional): Si True, normaliza el DataFrame. Si False, lo deja sin cambios.
        normalize_dict (dict, optional): Diccionario con valores predefinidos para normalizar otro DataFrame.
    
    Returns:
        Tuple[pd.DataFrame, Optional[dict]]: 
            - Si `normalize_dict` es proporcionado, solo devuelve el DataFrame normalizado.
            - Si no, devuelve el DataFrame normalizado junto con el diccionario de normalización.
    """
    
    normalized_df = df.copy()
    stats = None

    if 'area_units' in normalized_df.columns:
        normalized_df['area'] = normalized_df.apply(
            lambda row: sqft2m2(row['area'], reverse=False) if row['area_units'] == 'sqft' else row['area'], axis=1
        )
        # encode area_units
        normalized_df['north'] = normalized_df['area_units'].apply(lambda x: 1 if x == 'sqft' else 0)
        normalized_df = normalized_df.drop('area_units', axis=1)

    if normalize:
        if normalize_dict is not None:
            method = normalize_dict.get("method", method)

            for col, params in normalize_dict.items():
                if col in normalized_df.columns and isinstance(params, dict):
                    if method == 'gaussian':
                        mean = params.get("mean", 0)
                        std = params.get("std", 1)
                        normalized_df[col] = (normalized_df[col] - mean) / std
                    elif method == 'minmax':
                        min_val = params.get("min", 0)
                        max_val = params.get("max", 1)
                        normalized_df[col] = (normalized_df[col] - min_val) / (max_val - min_val)
                    else:
                        raise ValueError("El método de normalización debe ser 'minmax' o 'gaussian'.")

            return normalized_df, None
        
        else:
            stats = {"method": method}
            if method == 'minmax':
                for col in (columns or normalized_df.select_dtypes(include=['number']).columns):
                    min_val, max_val = normalized_df[col].min(), normalized_df[col].max()
                    normalized_df[col] = (normalized_df[col] - min_val) / (max_val - min_val)
                    stats[col] = {"min": min_val, "max": max_val}
            elif method == 'gaussian':
                for col in (columns or normalized_df.select_dtypes(include=['number']).columns):
                    mean, std = normalized_df[col].mean(), normalized_df[col].std()
                    normalized_df[col] = (normalized_df[col] - mean) / std
                    stats[col] = {"mean": mean, "std": std}
            else:
                raise ValueError("El método de normalización debe ser 'minmax' o 'gaussian'.")

    return normalized_df, stats
    
def denormalize_price(data: list, stats: dict) -> list:
    """Esta función desnormaliza los datos de precio de una lista.
    
    Args:
        data (list): La lista con los datos.
        stats (dict): Un diccionario con los valores para desnormalizar los datos del precio.
        
    Returns:
        list: La lista con los datos de precio desnormalizados.
    """ 
    method = stats.get("method")
    denormalized_data = []
    for value in data:
        if method == 'minmax':
            denormalized_value = value * (stats['price']['max'] - stats['price']['min']) + stats['price']['min']
        elif method == 'gaussian':
            denormalized_value = value * stats['price']['std'] + stats['price']['mean']
        else:
            raise ValueError("El método de normalización debe ser 'minmax' o 'gaussian'.")
        denormalized_data.append(denormalized_value)
    
    return denormalized_data

def denormalize_features(data: pd.DataFrame, stats: dict) -> pd.DataFrame:
    """Esta función desnormaliza las columnas de un DataFrame según el método especificado.
    
    Args:
        data (pd.DataFrame): El DataFrame con los datos.
        stats (dict): Un diccionario con los valores para desnormalizar los datos.
        
    Returns:
        pd.DataFrame: El DataFrame con las columnas desnormalizadas.
    """
    denormalized_df = data.copy()
    method = stats.get("method")
    
    for col, params in stats.items():
        if col != "method" and col in denormalized_df.columns:
            if method == 'minmax':
                min_val = params.get("min", 0)
                max_val = params.get("max", 1)
                denormalized_df[col] = data[col] * (max_val - min_val) + min_val
            elif method == 'gaussian':
                mean = params.get("mean", 0)
                std = params.get("std", 1)
                denormalized_df[col] = data[col] * std + mean
            else:
                raise ValueError("El método de normalización debe ser 'minmax' o 'gaussian'.")
    
    return denormalized_df

def fill_na(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    """Esta función rellena los valores nulos de las columnas especificadas con la media de la columna.

    Args:
        df (pd.DataFrame): Los datos.
        columns (List): Las columnas a rellenar.

    Returns:
        pd.DataFrame: Los datos con los valores nulos rellenados con la media.
    """
    for col in columns:
        df[col] = df[col].fillna(df[col].mean())
    return df

def add_features(df: pd.DataFrame, f_fill_na: bool = True, quantils = None) -> pd.DataFrame:
    """
    Agrega features derivadas al DataFrame.
    
    Args:
        df (pd.DataFrame): El DataFrame con los datos.
        f_fill_na (bool): Si es True, rellena los valores nulos en 'age' y 'rooms' con la media.
        quantils (tuple): Cuantiles para la creación de la columna 'size'.
        
    Returns:
        pd.DataFrame: El DataFrame con las nuevas features.
    """
    df, _ = normalize_df(df.copy(), normalize=False)  # Agrega columna 'north'

    if f_fill_na:
        df = fill_na(df, ['age', 'rooms'])

    # Features basadas en habitaciones
    df['rooms_per_area'] = df['rooms'] / df['area']
    # df['rooms_per_area_squared'] = df['rooms_per_area'] ** 2  # Podés revisar si aporta
    df['room_size'] = df['area'] / df['rooms']
    df['is_studio'] = (df['rooms'] == 1).astype(int)

    # Features de tamaño
    if quantils is not None:
        area_p25, area_p75 = quantils
    else:
        area_p25 = df['area'].quantile(0.25)
        area_p75 = df['area'].quantile(0.75)
    df['size'] = pd.cut(df['area'], bins=[-float('inf'), area_p25, area_p75, float('inf')], labels=['small', 'medium', 'large'])
    df = one_hot_encoder(df, 'size')

    # Interacción entre pileta y orientación
    df['pool_at_north'] = df['has_pool'] * df['north']

    # Features basadas en edad
    df['age_squared'] = df['age'] ** 2
    df['house_age'] = pd.cut(df['age'], bins=[0, 5, 15, float('inf')], labels=['new', 'medium', 'old'])
    df = one_hot_encoder(df, 'house_age')

    return df, (area_p25, area_p75)