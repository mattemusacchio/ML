from IPython.display import display, Markdown

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
