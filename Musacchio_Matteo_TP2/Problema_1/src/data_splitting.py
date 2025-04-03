import numpy as np

def train_val_split(df, test_size=0.2, random_state=95):

    """
    Divide el DataFrame en conjuntos de entrenamiento y validación sin usar sklearn.
    """
    np.random.seed(random_state)
    indices = np.random.permutation(len(df))
    split_idx = int(len(df) * (1 - test_size))
    train_indices, val_indices = indices[:split_idx], indices[split_idx:]
    train_df, val_df = df.iloc[train_indices], df.iloc[val_indices]
    return train_df, val_df