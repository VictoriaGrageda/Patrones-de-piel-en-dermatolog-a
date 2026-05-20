# Patrones de piel en dermatologia

Proyecto de Inteligencia Artificial 1 orientado al analisis de imagenes dermatologicas mediante vision por computadora y aprendizaje no supervisado. El sistema agrupa lesiones o patrones de piel por similitud visual usando caracteristicas de color, textura y forma.

> Este proyecto es academico. No reemplaza el diagnostico de un dermatologo ni debe usarse como herramienta medica final.
> El modelo se entrena desde cero con las imagenes colocadas en `data/raw/`. No se usan modelos preentrenados ni pesos externos.

## Objetivo

Construir un prototipo capaz de:

- Recibir imagenes dermatologicas.
- Preprocesar y normalizar las imagenes.
- Segmentar una region de interes aproximada.
- Extraer caracteristicas visuales interpretables.
- Entrenar el modelo de agrupamiento con la informacion propia del dataset.
- Reducir dimensionalidad con PCA.
- Agrupar patrones con K-Means, GMM, DBSCAN o Fuzzy C-Means.
- Evaluar la calidad de agrupamiento con metricas no supervisadas.
- Visualizar clusters y resultados.

## Area y subarea

- **Area:** Aprendizaje no supervisado.
- **Subarea:** Vision por computadora aplicada a dermatologia.

## Metodologia del sistema

1. **Entrada:** imagenes de piel en formato JPG, PNG, BMP o WEBP.
2. **Preprocesamiento:** correccion de orientacion, redimensionamiento, normalizacion y mejora de contraste.
3. **Segmentacion:** mascara aproximada de region de interes usando intensidad, color y operaciones morfologicas.
4. **Extraccion de caracteristicas:**
   - Color: estadisticas en RGB, HSV y LAB.
   - Textura: Local Binary Patterns e histograma HOG.
   - Forma: area, perimetro, excentricidad, solidez, extension y circularidad.
5. **Reduccion de dimensionalidad:** PCA para estabilizar el clustering y facilitar la visualizacion.
6. **Entrenamiento no supervisado:** el algoritmo aprende los grupos directamente desde las imagenes disponibles en `data/raw/`.
7. **Clustering:** K-Means, Gaussian Mixture Model, DBSCAN o Fuzzy C-Means.
8. **Evaluacion:** Silhouette Score, Davies-Bouldin Index y Calinski-Harabasz Index.
9. **Interpretacion:** tabla de imagenes agrupadas y mapa PCA en 2D.

## Estructura

```text
.
|-- app.py                         # Interfaz web con Streamlit
|-- train.py                       # Entrenamiento desde consola
|-- data/
|   |-- raw/                       # Imagenes dermatologicas de entrada
|   `-- processed/                 # Espacio para datos procesados
|-- docs/
|   `-- INFORME_PROYECTO.md        # Informe academico resumido
|-- models/                        # Modelos y transformadores guardados
|-- reports/                       # Resultados CSV
|-- src/
|   `-- skin_patterns/
|       |-- cli.py                 # Ejecucion por consola
|       |-- clustering.py          # Algoritmos y metricas
|       |-- config.py              # Configuracion general
|       |-- features.py            # Caracteristicas de imagen
|       |-- pipeline.py            # Flujo principal
|       `-- preprocessing.py       # Carga, contraste y segmentacion
|-- pyproject.toml
`-- requirements.txt
```

## Instalacion

Requisitos:

- Python 3.10 o superior.
- pip.
- Imagenes dermatologicas para prueba.

Crear entorno virtual e instalar dependencias:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Instalacion editable opcional:

```bash
pip install -e .
```

## Uso por consola

Coloca las imagenes en `data/raw/` y ejecuta:

```bash
python train.py --input data/raw --method kmeans --clusters 4
```

Otros metodos disponibles:

```bash
python train.py --input data/raw --method fuzzy --clusters 4
python train.py --input data/raw --method gmm --clusters 4
python train.py --input data/raw --method dbscan
```

Al finalizar se generan:

- `reports/clustering_results.csv`
- `models/skin_pattern_model.joblib`

Si aun no tienes imagenes reales, puedes crear un conjunto sintetico solo para probar la ejecucion. Estas imagenes no deben usarse como resultado final del proyecto:

```bash
python scripts/generate_sample_images.py
python train.py --input data/raw --method kmeans --clusters 3
```

## Entrenamiento con informacion propia

El entrenamiento se realiza usando unicamente las imagenes disponibles en la carpeta de entrada. El sistema no descarga pesos, no carga redes externas y no utiliza modelos previamente entrenados.

Para entrenar con informacion del grupo:

1. Colocar las imagenes recolectadas en `data/raw/`.
2. Ejecutar `python train.py --input data/raw --method kmeans --clusters 4`.
3. Revisar `reports/clustering_results.csv`.
4. Ajustar el numero de clusters y comparar metricas.

## Uso con interfaz web

```bash
streamlit run app.py
```

La app permite cargar varias imagenes, elegir el algoritmo, ajustar clusters y visualizar el mapa PCA junto con los resultados agrupados.

## Interpretacion de metricas

- **Silhouette Score:** mas alto suele indicar clusters mejor separados.
- **Davies-Bouldin Index:** mas bajo suele indicar mejor separacion.
- **Calinski-Harabasz Index:** mas alto suele indicar mejor estructura de clusters.

Estas metricas no prueban diagnostico clinico; solo miden coherencia estadistica del agrupamiento.

## Dataset del proyecto

Para cumplir con el requisito de entrenar con informacion propia, se debe trabajar con las imagenes recolectadas por el grupo o autorizadas para el proyecto. La carpeta final de entrenamiento debe ser:

```text
data/raw/
```

Recomendacion para organizar la informacion:

- Usar imagenes con buena iluminacion y enfoque.
- Evitar fotos duplicadas o borrosas.
- Mantener un criterio similar de captura para reducir ruido.
- Registrar en una hoja externa el origen de las imagenes y observaciones relevantes.

Los datasets publicos pueden mencionarse como referencia teorica, pero el entrenamiento entregable debe ejecutarse con la informacion definida por el grupo.

## Alcance actual

El proyecto implementa un prototipo funcional de analisis no supervisado. La clasificacion clinica supervisada no forma parte del alcance principal porque requiere etiquetas verificadas por especialistas, validacion medica y control de sesgos.

## Trabajo futuro

- Entrenar una CNN propia desde cero si se cuenta con suficientes imagenes etiquetadas.
- Agregar UMAP o t-SNE para visualizaciones alternativas.
- Comparar resultados con etiquetas clinicas si el dataset las incluye.
- Guardar reportes graficos en PDF.
- Mejorar segmentacion con modelos especializados para lesiones.
