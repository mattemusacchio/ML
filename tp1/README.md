# Trabajo Práctico 1: Regresión Lineal

Este repositorio contiene el código y los datos para el Trabajo Práctico 1 de la materia de Machine Learning, enfocado en la implementación y evaluación de modelos de regresión lineal.

## Estructura del Proyecto

```
|- data/                  # Carpeta para los datos del proyecto
   |- raw/                # Datos originales sin modificar
      |- casas_dev.csv    # Datos de desarrollo
      |- casas_test.csv   # Datos de prueba
   |- processed/          # Datos procesados y curados
|- src/                   # Carpeta para el código fuente del proyecto
   |- utils.py            # Funciones auxiliares
   |- metrics.py          # Funciones para calcular métricas
   |- preprocessing.py    # Funciones para el preprocesamiento
   |- models.py           # Clases para los modelos de ML
   |- data_splitting.py   # Funciones para dividir los datos
|- notebooks/             # Carpeta para Jupyter Notebooks
   |- Entrega_TP1.ipynb   # Respuestas de todos los ejercicios del TP
|- requirements.txt       # Especificar dependencias del proyecto
|- README.md              # Este archivo
```

## Requisitos

Para ejecutar el código de este proyecto, necesitarás tener instalado Python 3.8 o superior y las siguientes dependencias:

```
numpy==1.24.3
pandas==2.0.2
matplotlib==3.7.1
seaborn==0.12.2
scikit-learn==1.3.0
jupyter==1.0.0
notebook==6.5.4
```

Puedes instalar todas las dependencias con el siguiente comando:

```bash
pip install -r requirements.txt
```

## Uso

1. Clona este repositorio:
```bash
git clone <URL_DEL_REPOSITORIO>
cd <NOMBRE_DEL_REPOSITORIO>
```

2. Instala las dependencias:
```bash
pip install -r requirements.txt
```

3. Abre el notebook principal para ver el análisis completo:
```bash
jupyter notebook notebooks/Entrega_TP1.ipynb
```

## Descripción de los Módulos

- **utils.py**: Contiene funciones auxiliares como `save_results()` para guardar resultados y `load_model()` para cargar modelos guardados.
- **metrics.py**: Implementa funciones para calcular métricas de evaluación como `mse()`, `mae()` y `rmse()`.
- **preprocessing.py**: Contiene funciones para el preprocesamiento de datos como `one_hot_encoder()`, `normalize()` y `handle_missing_values()`.
- **models.py**: Implementa la clase `LinearRegression()` para el modelo de regresión lineal.
- **data_splitting.py**: Contiene funciones para dividir los datos como `train_val_split()` y `cross_val()`.

## Autores

- Nombre Apellido 1 (Padrón: XXXXX)
- Nombre Apellido 2 (Padrón: XXXXX)
- Nombre Apellido 3 (Padrón: XXXXX) 