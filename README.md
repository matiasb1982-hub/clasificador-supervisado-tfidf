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

- `max_features
