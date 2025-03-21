import pandas as pd
from models import LinearRegression
from IPython.display import display, Markdown

def pretty_print_df(df: pd.DataFrame, index=False, float_format=None):
    df = df.rename(columns=lambda x: x.replace('_', ' ').title() if isinstance(x, str) else x)
    # apply float format to all columns if it is type float
    if float_format is not None:
        for col in df.select_dtypes(include=['float']).columns:
            df[col] = df[col].map(lambda x: f"{x:.2f}" if not pd.isnull(x) else x)
    display(Markdown(df.to_markdown(index=index)))

def analyze_null_values(df):
    """Analyzes the number of null values in the DataFrame."""
    null_counts = df.isnull().sum()
    null_percentages = (null_counts / len(df)) * 100
    null_summary = pd.DataFrame({
        'Null Count': null_counts,
        'Percentage': null_percentages
    }).sort_values(by='Null Count', ascending=False)

    num_null_rows = df.isnull().any(axis=1).sum()
    print(f"Number of rows with at least one null value: {num_null_rows}\n")
    print("Number of null values per column and their percentage:")
    pretty_print_df(null_summary, index=True, float_format='%.2f')

def show_unique_values(df):
    """Displays unique values for columns that are not of type float."""
    non_float_columns = df.select_dtypes(exclude=['float']).columns
    unique_values = {col: df[col].unique() for col in non_float_columns}
    
    print("\nUnique values for columns that are not of type float:")
    for col, values in unique_values.items():
        print(f"{col}: {values}")

def analyze_dataframe_size(df):
    """Displays information about the DataFrame size with and without null values."""
    original_length = len(df)
    df_dropna_length = len(df.dropna())
    percentage_difference = ((original_length - df_dropna_length) / original_length) * 100

    print(f"Original DataFrame length: {original_length}")
    print(f"DataFrame length without null values: {df_dropna_length}")
    print(f"Percentage difference after removing nulls: {percentage_difference:.2f}%")
    
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
        
def remove_columns(columns, to_remove):
    """
    Elimina columnas específicas de un conjunto de columnas de un DataFrame.
    """
    if isinstance(to_remove, str):
        to_remove = [to_remove]
    return [col for col in columns if col not in to_remove]


def process_binary_columns(df, exclude_column='price', binary_columns=None):
    """
    Identifica columnas binarias y las convierte a enteros (0 y 1).
    Si se pasan columnas binarias específicas, simplemente las convierte.
    """

    def convert_columns_to_int(df, columns):
        df_copy = df.copy()
        for col in columns:
            df_copy[col] = df_copy[col].astype(int)
        return df_copy

    if binary_columns is not None:
        df_binary_int = convert_columns_to_int(df, binary_columns)
        return None, None, df_binary_int

    binary_columns = []
    non_binary_columns = []

    for col in df.columns:
        values = df[col].dropna().unique()
        if set(values).issubset({0, 1}) or set(values).issubset({True, False}):
            binary_columns.append(col)
        elif exclude_column is None or col != exclude_column:
            non_binary_columns.append(col)

    df_binary_int = convert_columns_to_int(df, binary_columns)
    return non_binary_columns, binary_columns, df_binary_int