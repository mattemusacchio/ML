import numpy as np
import pandas as pd
from typing import Tuple

def train_val_split(df: pd.DataFrame, val_pct: float, seed = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Esta función separa el DataFrame en dos partes: entrenamiento y validación según un porcentaje dado.

    Args:
        df (pd.DataFrame): Los datos a separar
        val_pct (float): El porcentaje de datos que queremos que vayan al dataframe de Validación.
        seed (int, optional): La semilla para el generador de números aleatorios. Defaults to None.
        
    Returns:
        pd.DataFrame: El dataframe de entrenamiento
        pd.DataFrame: El dataframe de validación
    """
    
    if seed:
        np.random.seed(seed)
    
    n_val = int(len(df) * val_pct)
    val_indices = np.random.choice(df.index, size=n_val, replace=False)
    df_val = df.loc[val_indices]
    df_train = df.drop(val_indices)
    
    return df_train, df_val