import json

def normalize_train_data(df, cols_to_normalize, save_path):
    """
    Normaliza las columnas indicadas usando Min-Max Scaling y guarda los parámetros.
    """
    normalization_params = {}

    for col in cols_to_normalize:
        col_min = float(df[col].min()) 
        col_max = float(df[col].max())  
        df[col] = (df[col] - col_min) / (col_max - col_min)
        normalization_params[col] = {"min": col_min, "max": col_max}


    with open(save_path, 'w') as f:
        json.dump(normalization_params, f, indent=4)

    return df

def normalize_validation_data(val_data, normalization_params_path):
    """
    Normaliza el dataset de validación utilizando los parámetros guardados del entrenamiento.

    Parámetros:
    - val_data: DataFrame, conjunto de datos de validación.
    - normalization_params_path: str, ruta del JSON con los parámetros de normalización del entrenamiento.

    Retorna:
    - val_data_normalized: DataFrame, conjunto de datos de validación normalizado.
    """
   
    with open(normalization_params_path, 'r') as f:
        normalization_params = json.load(f)

    
    for col in normalization_params.keys():
        if col in val_data.columns:
            col_min = normalization_params[col]["min"]
            col_max = normalization_params[col]["max"]
            val_data[col] = (val_data[col] - col_min) / (col_max - col_min)

    return val_data


def inverse_normalize_data(y_normalized, col, params_path):
    """
    Desnormaliza los valores de salida usando los parámetros de normalización guardados en JSON.

    Parámetros:
    - y_normalized: valores normalizados.
    - col: string con el nombre de la columna a desnormalizar.
    - params_path: ruta del archivo donde se guardadon los parámetros de la normalización.
    """

    with open(params_path, 'r') as f:
        normalization_params = json.load(f)

    col_min = normalization_params[col]['min']
    col_max = normalization_params[col]['max']

    y_denormalized = y_normalized * (col_max - col_min) + col_min

    return y_denormalized