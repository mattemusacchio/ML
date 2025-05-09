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
import itertools
import numpy as np

def grid_search_model_M1(param_grid, X_train, y_train, X_val, y_val, epochs=100, early_stopping=True, patience=10):
    from .models import NeuralNetwork, linear_schedule, exponential_schedule
    keys = list(param_grid.keys())
    values = list(param_grid.values())
    combinations = list(itertools.product(*values))

    best_config = None
    best_loss = float('inf')
    best_history = None

    print(f"🔍 Probing {len(combinations)} configurations...")

    for i, combo in enumerate(combinations):
        config = dict(zip(keys, combo))
        print(f"\n🔧 Running config {i+1}/{len(combinations)}: {config}")

        # Elegir scheduler
        if config["scheduler"] == "linear":
            scheduler_fn = lambda epoch: linear_schedule(epoch, initial_lr=config["lr"], final_lr=0.001, saturate_epoch=epochs)
        elif config["scheduler"] == "exponential":
            scheduler_fn = exponential_schedule(initial_lr=config["lr"], gamma=0.95)
        else:
            scheduler_fn = None

        model = NeuralNetwork(
            layer_sizes=[X_train.shape[1]] + config["hidden_layers"] + [y_train.shape[1]],
            learning_rate=config["lr"],
            l2_lambda=config["lambda_reg"],
            dropout_rate=config["dropout"],
            use_adam=config["use_adam"]
        )

        history = model.train(
            X_train, y_train,
            X_val, y_val,
            epochs=epochs,
            verbose=False,
            batch_size=config["batch_size"],
            use_scheduler=config["scheduler"] is not None,
            scheduler_fn=scheduler_fn,
            early_stopping=early_stopping,
            patience=patience
        )

        val_loss = history["val_loss"][-1]
        print(f"✅ Final Val Loss: {val_loss:.4f}")

        if val_loss < best_loss:
            best_loss = val_loss
            best_config = config
            best_history = history

    print("\n🏆 Best config:", best_config)
    print(f"📉 Best validation loss: {best_loss:.4f}")
    return best_config, best_history
