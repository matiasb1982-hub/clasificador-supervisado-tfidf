# Clasificador Supervisado de Noticias con TF-IDF

## Descripción

Proyecto de clasificación supervisada de noticias de AG News en cuatro categorías:
Business, Sci_Tech, Sports y World.

La solución utiliza preprocesamiento de texto, lematización, vectorización TF-IDF
y Regresión Logística para predecir la categoría de cada noticia.

## Dataset

Se utilizaron los archivos del Módulo 2:

- `ag_news_train.csv`: 8.000 noticias.
- `ag_news_test.csv`: 2.000 noticias.

Ambos archivos contienen las columnas `text` y `label`. Las cuatro categorías
están representadas de forma equilibrada en ambos conjuntos.

## Preprocesamiento

El texto se normaliza convirtiéndolo a minúsculas, decodificando entidades HTML,
eliminando etiquetas HTML, URLs, correos electrónicos y caracteres no relevantes.
Luego se utiliza spaCy para lematizar los tokens.

Se conservaron las etiquetas originales del dataset.

## Experimentación con TF-IDF

Se compararon seis configuraciones usando una división interna del conjunto de
entrenamiento. La selección se realizó según F1 macro, utilizando accuracy como
criterio secundario.

| max_features | ngram_range | Accuracy de validación | F1 macro |
|---:|:---:|---:|---:|
| 10.000 | (1, 1) | 0,8925 | 0,8920 |
| 20.000 | (1, 1) | 0,8925 | 0,8920 |
| 5.000 | (1, 1) | 0,8900 | 0,8895 |
| 10.000 | (1, 2) | 0,8831 | 0,8825 |
| 20.000 | (1, 2) | 0,8831 | 0,8824 |
| 5.000 | (1, 2) | 0,8812 | 0,8806 |

La configuración seleccionada fue:

- `max_features=10000`
- `ngram_range=(1, 1)`
- `min_df=2`
- `sublinear_tf=True`

En estas pruebas, los unigramas obtuvieron mejores resultados que la combinación
de unigramas y bigramas. Aumentar el límite a 20.000 no mejoró el resultado
porque el vocabulario disponible en esa configuración era menor.

## Modelo elegido

Se eligió Regresión Logística como clasificador de referencia. Es un modelo
supervisado adecuado para clasificación multiclase y funciona bien con las
matrices dispersas generadas por TF-IDF. Además, permite establecer una línea
base clara, eficiente y reproducible.

La configuración utilizada fue `max_iter=1000` y `random_state=42`.

## Prevención de data leakage

La selección de parámetros se realizó mediante una división interna del conjunto
de entrenamiento. El vectorizador se ajustó exclusivamente con los textos de
entrenamiento de cada prueba de validación.

Una vez elegida la configuración, se ajustó el vectorizador con los 8.000 textos
de entrenamiento y se aplicó `transform` a los 2.000 textos de prueba. El
conjunto de prueba no se utilizó para elegir los parámetros.

## Resultados finales en el conjunto de prueba

- **Accuracy:** 88,30 %
- **Precision macro:** 88,28 %
- **Recall macro:** 88,30 %
- **F1 macro:** 88,27 %
- **Noticias de prueba:** 2.000
- **Características TF-IDF finales:** 9.690

### Métricas por categoría

| Categoría | Precision | Recall | F1-score | Cantidad |
|---|---:|---:|---:|---:|
| Business | 0,8446 | 0,8480 | 0,8463 | 500 |
| Sci_Tech | 0,8594 | 0,8560 | 0,8577 | 500 |
| Sports | 0,9284 | 0,9600 | 0,9440 | 500 |
| World | 0,8986 | 0,8680 | 0,8830 | 500 |

## Análisis de errores

La categoría con mayor dificultad fue Business, con un 15,2 % de error.
Le siguió Sci_Tech, con un 14,4 %. World tuvo un 13,2 % de error, mientras
que Sports presentó solo un 4 %.

Las confusiones más frecuentes fueron:

- 44 noticias de Sci_Tech clasificadas como Business.
- 43 noticias de Business clasificadas como Sci_Tech.
- 28 noticias de World clasificadas como Business.
- 26 noticias de Business clasificadas como World.

La confusión entre Business y Sci_Tech puede explicarse porque ambas categorías
comparten vocabulario relacionado con empresas, productos, mercados e innovación.
Sports se distingue mejor por su vocabulario específico.

## Cómo ejecutar el proyecto

1. Abrir `Clasificador_Supervisado_TFIDF_Final.ipynb` en Google Colab.
2. Instalar las dependencias indicadas en `requirements.txt`.
3. Cargar `ag_news_train.csv` y `ag_news_test.csv`.
4. Ejecutar las celdas en orden.
5. Revisar el reporte de clasificación y la matriz de confusión.

## Tecnologías

- Python
- pandas
- scikit-learn
- spaCy
- Matplotlib
- Seaborn
- TF-IDF
- Regresión Logística

