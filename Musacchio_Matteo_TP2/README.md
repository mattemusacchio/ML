# Trabajo Práctico: Clasificación Binaria

Este repositorio contiene el desarrollo completo del Trabajo Práctico de Aprendizaje Automático, centrado en la implementación, entrenamiento y evaluación de modelos de clasificación para un conjunto de datos binario.

## Descripción del Trabajo Práctico

El objetivo principal de este trabajo práctico es comparar distintas técnicas de clasificación supervisada sobre un conjunto de datos real. Se busca evaluar la capacidad de generalización de cada modelo a través de distintas métricas y visualizaciones, así como realizar una interpretación detallada de sus resultados.

En este proyecto:

- Entrenamos y evaluamos tres modelos de clasificación: **LDA**, **Regresión Logística** y **Random Forest**
- Se implementó desde cero el clasificador de regresión logística
- Se trabajó sobre un conjunto de datos binario preprocesado y balanceado
- Se analizaron las métricas estándar: Accuracy, Precisión, Recall, F1-Score, AUC-ROC y AUC-PR
- Se generaron visualizaciones como curvas ROC, curvas PR y matrices de confusión
- Se utilizaron conjuntos separados para entrenamiento/validación y testeo

Los resultados muestran que todos los modelos logran un buen desempeño, destacándose el modelo de Random Forest como el más robusto en todas las métricas.


Para instalar todas las dependencias, ejecutar:

```bash
pip install -r requirements.txt
