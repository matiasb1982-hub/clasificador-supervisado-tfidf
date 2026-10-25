# Clasificador Supervisado de Noticias con TF-IDF

## 1. Descripción del proyecto

Este proyecto implementa un clasificador supervisado de noticias utilizando técnicas de procesamiento de lenguaje natural (NLP) y Machine Learning clásico.

El objetivo es clasificar noticias del dataset AG News en cuatro categorías: Business, Sci_Tech, Sports y World, a partir del contenido textual.

El proyecto forma parte de la preentrega de clasificación supervisada con TF-IDF.

## 2. Dataset

Se utilizó el corpus AG News, con los siguientes conjuntos:

- Entrenamiento: 8.000 noticias.
- Prueba: 2.000 noticias.
- Categorías: Business, Sci_Tech, Sports y World.

Los conjuntos de entrenamiento y prueba se mantuvieron separados durante la vectorización y la evaluación.

## 3. Preprocesamiento del texto

Se reutilizaron las técnicas de preprocesamiento trabajadas en el módulo anterior:

- Limpieza y normalización del texto.
- Lematización.
- Eliminación de stop-words.
- Preparación del texto para su representación numérica.

Para el modelo se utilizó la columna `text_lemmas_no_stop`.

## 4. Vectorización TF-IDF

Se utilizó `TfidfVectorizer` de Scikit-learn para transformar los documentos en vectores numéricos ponderados según la importancia de sus términos.

Se compararon tres configuraciones:

| max_features | ngram_range | Accuracy |
|---:|:---:|---:|
| 10.000 | (1,1) | 89,85 % |
| 20.000 | (1,2) | 89,35 % |
| 50.000 | (1,2) | 89,30 % |

La configuración seleccionada fue la de 10.000 características y unigramas, por haber obtenido la mayor exactitud en el conjunto de prueba.

Los parámetros del modelo definitivo fueron:

`max_features=10000`: limita la cantidad máxima de características.

- `ngram_range=(1,1)`: utiliza unigramas, es decir, palabras individuales.
- `min_df=2`: descarta términos presentes en menos de dos documentos del entrenamiento.
- `sublinear_tf=True`: aplica una escala logarítmica a la frecuencia de los términos.

Para evitar *data leakage*, el vectorizador se ajustó únicamente con los datos de entrenamiento mediante `fit_transform`. Para el conjunto de prueba se utilizó `transform`.

## 5. Modelo de clasificación

Se seleccionó Regresión Logística (`LogisticRegression`) como modelo de referencia por su utilidad en problemas de clasificación de texto y su compatibilidad con matrices dispersas generadas por TF-IDF.

Parámetros principales:

- `max_iter=1000`
- `random_state=42`

## 6. Resultados de la evaluación

El modelo obtuvo los siguientes resultados sobre el conjunto de prueba:

- **Accuracy:** 89,85 %.
- **F1-score macro:** aproximadamente 0,90.
- **Cantidad de noticias evaluadas:** 2.000.

### Resultados por categoría

| Categoría | Precision | Recall | F1-score |
|---|---:|---:|---:|
| Business | 0,86 | 0,86 | 0,86 |
| Sci_Tech | 0,87 | 0,88 | 0,87 |
| Sports | 0,96 | 0,97 | 0,96 |
| World | 0,91 | 0,89 | 0,90 |

## 7. Análisis de errores

La matriz de confusión mostró que **Sports fue la categoría con mejor desempeño**, con 483 noticias correctamente clasificadas de 500.

Las principales confusiones se produjeron entre Business y Sci_Tech:

- 41 noticias de Business fueron clasificadas como Sci_Tech.
- 42 noticias de Sci_Tech fueron clasificadas como Business.

También hubo confusiones entre World y Business. Estas dificultades pueden relacionarse con el vocabulario compartido entre categorías.

## 8. Tecnologías y dependencias

El proyecto se desarrolló en Google Colab con Python 3.13.16.

Versiones registradas:

- pandas 2.2.3
- scikit-learn 1.6.1
- matplotlib 3.10.0
- seaborn 0.13.2
- nltk 3.9.1

Las dependencias están especificadas en `requirements.txt`.

## 9. Estructura del repositorio

- `EDA_NLP_BoccoMatias_ipynb_.ipynb`: notebook con el desarrollo del proyecto.
- `requirements.txt`: versiones de las librerías utilizadas.
- `README.md`: documentación del proyecto.

## 10. Conclusiones

La combinación de TF-IDF y Regresión Logística permitió clasificar noticias en cuatro categorías con una exactitud del 89,85 %.

La comparación de configuraciones mostró que aumentar la cantidad de características e incorporar bigramas no mejoró el resultado en este experimento.

Como futuras mejoras, se propone evaluar otros algoritmos de clasificación, ajustar hiperparámetros y explorar representaciones alternativas del texto.

El proyecto integra preparación de datos, vectorización, entrenamiento, evaluación y análisis de errores propios de un proceso de clasificación supervisada.
