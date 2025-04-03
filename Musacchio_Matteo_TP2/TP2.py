# Trabajo Práctico 2: Clasificación y Ensemble Learning
# Autor: Matteo Musacchio

# Importamos las librerías necesarias
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from IPython.display import display, Markdown
from Problema_1.src.utils import pretty_print_df, handle_missing_values, one_hot_encoding

# Configuración para visualizaciones
plt.style.use('seaborn-v0_8-whitegrid')  # Actualizado para versiones recientes de seaborn
sns.set(style="whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12

# Definimos la ruta base del proyecto
BASE_PATH = r'c:\Users\Matteo\Documents\Facultad\tercero\primercuatri\ML\ML\Musacchio_Matteo_TP2'

# Creamos las carpetas necesarias si no existen
os.makedirs(os.path.join(BASE_PATH, 'Problema 1', 'data', 'processed'), exist_ok=True)
os.makedirs(os.path.join(BASE_PATH, 'Problema 1', 'data', 'raw'), exist_ok=True)
os.makedirs(os.path.join(BASE_PATH, 'Problema 2', 'data', 'raw'), exist_ok=True)
os.makedirs(os.path.join(BASE_PATH, 'Problema 2', 'data', 'processed'), exist_ok=True)

# Problema 1: Diagnóstico de Cáncer de Mama
display(Markdown("## Problema 1: Diagnóstico de Cáncer de Mama"))

# 1.1 Análisis Exploratorio de Datos
cell_balanced_dev = pd.read_csv(os.path.join(BASE_PATH, 'Problema 1', 'data', 'raw', 'cell_diagnosis_dev.csv'))
cell_balanced_test = pd.read_csv(os.path.join(BASE_PATH, 'Problema 1', 'data', 'raw', 'cell_diagnosis_test.csv'))
cell_imbalanced_dev = pd.read_csv(os.path.join(BASE_PATH, 'Problema 1', 'data', 'raw', 'cell_diagnosis_dev_imbalanced.csv'))
cell_imbalanced_test = pd.read_csv(os.path.join(BASE_PATH, 'Problema 1', 'data', 'raw', 'cell_diagnosis_test_imbalanced.csv'))

# Comenzamos con el análisis del dataset
display(Markdown("### Análisis del dataset de desarrollo"))

# Información general del dataset
display(Markdown("#### Información general del dataset"))
pretty_print_df(pd.DataFrame({
    'Columnas': cell_balanced_dev.columns,
    'Tipo de dato': cell_balanced_dev.dtypes,
    'No nulos': cell_balanced_dev.count()
}), title="Información del dataset")

# Estadísticas descriptivas
display(Markdown("#### Estadísticas descriptivas"))
pretty_print_df(cell_balanced_dev.describe(), title="Estadísticas descriptivas")

# Verificamos valores faltantes
display(Markdown("#### Valores faltantes por columna"))
missing_values = cell_balanced_dev.isnull().sum()
missing_percent = (missing_values / len(cell_balanced_dev)) * 100
missing_data = pd.DataFrame({'Missing Values': missing_values, 'Percentage': missing_percent})
pretty_print_df(missing_data[missing_data['Missing Values'] > 0].sort_values('Missing Values', ascending=False), 
               title="Valores faltantes")

# Verificamos la distribución de la variable objetivo (Diagnosis)
display(Markdown("#### Distribución de la variable objetivo (Diagnosis)"))
target_distribution = cell_balanced_dev['Diagnosis'].value_counts()
pretty_print_df(pd.DataFrame({
    'Diagnosis': target_distribution.index,
    'Cantidad': target_distribution.values,
    'Porcentaje': (target_distribution.values / len(cell_balanced_dev) * 100).round(2)
}), title="Distribución del diagnóstico")

# Visualizamos la distribución de la variable objetivo
plt.figure(figsize=(10, 6))
ax = sns.countplot(x='Diagnosis', data=cell_balanced_dev)
plt.title('Distribución de la variable objetivo (Diagnosis)')
plt.xlabel('Diagnóstico (0: Benigno, 1: Maligno)')
plt.ylabel('Cantidad de muestras')

# Añadimos etiquetas con los valores exactos
for p in ax.patches:
    ax.annotate(f'{p.get_height()}', (p.get_x() + p.get_width() / 2., p.get_height()),
                ha='center', va='bottom', fontsize=12)

# Analizamos la distribución de las variables categóricas
categorical_features = ['CellType', 'GeneticMutation']
display(Markdown("#### Distribución de variables categóricas"))

for feature in categorical_features:
    plt.figure(figsize=(10, 6))
    ax = sns.countplot(x=feature, data=cell_balanced_dev, hue='Diagnosis')
    plt.title(f'Distribución de {feature} por diagnóstico')
    plt.xlabel(feature)
    plt.ylabel('Cantidad de muestras')
    
    # Añadimos etiquetas con los valores exactos
    for p in ax.patches:
        ax.annotate(f'{p.get_height()}', (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='bottom', fontsize=10)

# Analizamos la distribución de las variables numéricas
numeric_features = cell_balanced_dev.select_dtypes(include=['float64', 'int64']).columns.tolist()
if 'Diagnosis' in numeric_features:
    numeric_features.remove('Diagnosis')

# Creamos histogramas para cada variable numérica
plt.figure(figsize=(20, 15))
for i, feature in enumerate(numeric_features[:12], 1):  # Limitamos a 12 features por figura
    plt.subplot(4, 3, i)
    sns.histplot(cell_balanced_dev[feature], kde=True)
    plt.title(f'Distribución de {feature}')
plt.tight_layout()

# Si hay más de 12 features, creamos otra figura
if len(numeric_features) > 12:
    plt.figure(figsize=(20, 15))
    for i, feature in enumerate(numeric_features[12:], 1):
        plt.subplot(4, 3, i)
        sns.histplot(cell_balanced_dev[feature], kde=True)
        plt.title(f'Distribución de {feature}')
    plt.tight_layout()

# Identificamos posibles outliers usando boxplots
plt.figure(figsize=(20, 15))
for i, feature in enumerate(numeric_features[:12], 1):
    plt.subplot(4, 3, i)
    sns.boxplot(y=cell_balanced_dev[feature])
    plt.title(f'Boxplot de {feature}')
plt.tight_layout()

# Si hay más de 12 features, creamos otra figura para boxplots
if len(numeric_features) > 12:
    plt.figure(figsize=(20, 15))
    for i, feature in enumerate(numeric_features[12:], 1):
        plt.subplot(4, 3, i)
        sns.boxplot(y=cell_balanced_dev[feature])
        plt.title(f'Boxplot de {feature}')
    plt.tight_layout()

# Analizamos la correlación entre variables numéricas
display(Markdown("#### Matriz de correlación"))
correlation_matrix = cell_balanced_dev[numeric_features].corr()

# Visualizamos la matriz de correlación
plt.figure(figsize=(16, 14))
mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm', linewidths=0.5, mask=mask)
plt.title('Matriz de correlación')

# Identificamos las variables más correlacionadas con el target
if 'Diagnosis' in cell_balanced_dev.columns:
    # Calculamos la correlación de cada feature con el target
    target_correlations = cell_balanced_dev[numeric_features + ['Diagnosis']].corr()['Diagnosis'].sort_values(ascending=False)
    display(Markdown("#### Correlación de las variables con el target (Diagnosis)"))
    pretty_print_df(target_correlations, title="Correlación de las variables con el target (Diagnosis)")

    # Visualizamos las 10 variables más correlacionadas con el target (o todas si hay menos de 10)
    plt.figure(figsize=(12, 8))
    top_correlations = target_correlations[1:min(11, len(target_correlations))]  # Excluimos la correlación del target consigo mismo
    sns.barplot(x=top_correlations.values, y=top_correlations.index)
    plt.title('Top variables más correlacionadas con el diagnóstico')
    plt.xlabel('Coeficiente de correlación')

# Visualizamos la relación entre las variables más correlacionadas y el target
if 'Diagnosis' in cell_balanced_dev.columns:
    top_features = target_correlations[1:6].index.tolist()  # Top 5 features
    
    plt.figure(figsize=(15, 10))
    for i, feature in enumerate(top_features, 1):
        plt.subplot(2, 3, i)
        sns.boxplot(x='Diagnosis', y=feature, data=cell_balanced_dev)
        plt.title(f'{feature} vs Diagnóstico')
        plt.xlabel('Diagnóstico (0: Benigno, 1: Maligno)')
    plt.tight_layout()

# Análisis de valores extremos (outliers)
display(Markdown("#### Detección de outliers en variables numéricas"))
outliers_data = []
for feature in numeric_features:
    Q1 = cell_balanced_dev[feature].quantile(0.25)
    Q3 = cell_balanced_dev[feature].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = ((cell_balanced_dev[feature] < lower_bound) | (cell_balanced_dev[feature] > upper_bound)).sum()
    if outliers > 0:
        outliers_data.append({
            'Variable': feature,
            'Outliers': outliers,
            'Porcentaje': f"{(outliers/len(cell_balanced_dev)*100):.2f}%"
        })

pretty_print_df(pd.DataFrame(outliers_data), title="Detección de outliers")

display(Markdown("Análisis exploratorio completado. Se han generado visualizaciones en la carpeta del proyecto."))

##############################################################################################################################################

display(Markdown("## Entrenamiento y Evaluación del Modelo de Regresión Logística"))

# Importamos las clases y funciones necesarias
from Problema_1.src.models import LogisticRegression
from Problema_1.src.data_splitting import train_val_split
from Problema_1.src.utils import pretty_print_df, handle_missing_values, one_hot_encoding

# Separamos features y target
X = cell_balanced_dev.drop('Diagnosis', axis=1)
y = cell_balanced_dev['Diagnosis']

# Debugging: Verificar datos
display(Markdown("### Verificación de datos"))
display(Markdown("#### Shape de los datos"))
print(f"X shape: {X.shape}")
print(f"y shape: {y.shape}")

# Verificar tipos de datos
display(Markdown("#### Tipos de datos"))
print(f"Tipos de X:\n{X.dtypes}")
print(f"Tipo de y: {y.dtype}")

# Manejamos valores faltantes
display(Markdown("#### Manejo de valores faltantes"))
X = handle_missing_values(X)
y = handle_missing_values(pd.DataFrame(y))

# Aplicamos one-hot encoding a las variables categóricas
display(Markdown("#### Aplicando one-hot encoding"))
categorical_columns = X.select_dtypes(include=['object', 'category']).columns.tolist()
print(f"Columnas categóricas: {categorical_columns}")
X = one_hot_encoding(X, categorical_columns)
print(f"Nuevo shape de X después de one-hot encoding: {X.shape}")

# Guardamos los datasets procesados para análisis
display(Markdown("#### Guardando datasets procesados"))

# Creamos un DataFrame con X y y juntos para guardar
processed_data = pd.DataFrame(X, columns=X.columns)
processed_data['Diagnosis'] = y

# Guardamos en la carpeta data/processed
processed_data.to_csv('Problema_1/data/processed/cell_diagnosis_processed.csv', index=False)

print("Dataset procesado guardado en: Problema_1/data/processed/cell_diagnosis_processed.csv")


# Dividimos los datos en train y validation
X_train, X_val = train_val_split(X, test_size=0.2)
y_train, y_val = train_val_split(y, test_size=0.2)

# Debugging: Verificar split
display(Markdown("#### Shape después del split"))
print(f"X_train shape: {X_train.shape}")
print(f"X_val shape: {X_val.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"y_val shape: {y_val.shape}")

# Convertimos a arrays de numpy y nos aseguramos de que sean float64
X_train = X_train.values.astype(np.float64)
X_val = X_val.values.astype(np.float64)
y_train = y_train.values.astype(np.float64)
y_val = y_val.values.astype(np.float64)

# Debugging: Verificar arrays numpy
display(Markdown("#### Verificación de arrays numpy"))
print(f"X_train dtype: {X_train.dtype}")
print(f"y_train dtype: {y_train.dtype}")
print(f"Valores únicos en y_train: {np.unique(y_train)}")

# Definimos los valores de lambda para probar
lambda_values = [0.001, 0.01, 0.1, 1.0, 10.0]
best_lambda = None
best_f1 = 0

# Buscamos el mejor valor de lambda usando el conjunto de validación
display(Markdown("### Búsqueda del mejor valor de lambda"))
results = []

for lambda_reg in lambda_values:
    # Entrenamos el modelo
    model = LogisticRegression(lambda_reg=lambda_reg)
    model.fit(X_train, y_train)
    
    # Evaluamos en el conjunto de validación
    metrics = model.evaluate(X_val, y_val)
    
    results.append({
        'Lambda': lambda_reg,
        'F1-Score': metrics['f1'],
        'Accuracy': metrics['accuracy'],
        'Precision': metrics['precision'],
        'Recall': metrics['recall'],
        'AUC-ROC': metrics['auc_roc'],
        'AUC-PR': metrics['auc_pr']
    })
    
    if metrics['f1'] > best_f1:
        best_f1 = metrics['f1']
        best_lambda = lambda_reg

# Mostramos los resultados
pretty_print_df(pd.DataFrame(results), title="Resultados para diferentes valores de lambda")

# Entrenamos el modelo final con el mejor lambda
display(Markdown("### Modelo Final"))
final_model = LogisticRegression(lambda_reg=0.001)
final_model.fit(X_train, y_train)

# Debugging: Verificar pesos del modelo
display(Markdown("#### Verificación de pesos del modelo"))
print(f"Shape de weights: {final_model.weights.shape}")
print(f"Bias: {final_model.bias}")

# Evaluamos el modelo final
final_metrics = final_model.evaluate(X_val, y_val)

# Mostramos las métricas finales
display(Markdown("#### Métricas del Modelo Final"))
metrics_df = pd.DataFrame({
    'Métrica': ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC-ROC', 'AUC-PR'],
    'Valor': [
        final_metrics['accuracy'],
        final_metrics['precision'],
        final_metrics['recall'],
        final_metrics['f1'],
        final_metrics['auc_roc'],
        final_metrics['auc_pr']
    ]
})
pretty_print_df(metrics_df, title="Métricas del modelo final")

# Mostramos la matriz de confusión
display(Markdown("#### Matriz de Confusión"))
conf_matrix = pd.DataFrame(
    final_metrics['confusion_matrix'],
    columns=['Predicción Negativa', 'Predicción Positiva'],
    index=['Real Negativa', 'Real Positiva']
)
pretty_print_df(conf_matrix, title="Matriz de confusión")

# Graficamos las curvas ROC y PR
display(Markdown("#### Curvas de Evaluación"))
final_model.plot_curves(X_val, y_val)

##############################################################################################################################################
