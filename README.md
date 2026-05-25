# Patrones de piel en dermatologia

Proyecto de Inteligencia Artificial 1 orientado al analisis de imagenes dermatologicas mediante vision por computadora. El repositorio conserva un flujo exploratorio no supervisado y agrega entrenamiento supervisado con HAM10000 usando una CNN propia entrenada desde cero.

> Este proyecto es academico. No reemplaza el diagnostico de un dermatologo ni debe usarse como herramienta medica final.
> El modelo HAM10000 se entrena desde cero con las imagenes locales del dataset. No se usan modelos preentrenados, backbones externos ni pesos descargados.

## Objetivo

Construir un prototipo capaz de:

- Recibir imagenes dermatologicas.
- Preprocesar y normalizar las imagenes.
- Segmentar una region de interes aproximada.
- Extraer caracteristicas visuales interpretables.
- Entrenar el modelo de agrupamiento con la informacion propia del dataset.
- Entrenar una CNN propia para clasificar las 7 clases de HAM10000 desde inicializacion aleatoria.
- Reducir dimensionalidad con PCA.
- Agrupar patrones con K-Means, GMM, DBSCAN o Fuzzy C-Means.
- Evaluar la calidad de agrupamiento con metricas no supervisadas.
- Evaluar clasificacion supervisada con accuracy, macro F1, matriz de confusion y reporte por clase.
- Visualizar clusters y resultados.

## Area y subarea

- **Area:** Vision por computadora y aprendizaje automatico.
- **Subarea:** Vision por computadora aplicada a dermatologia.

## Dataset HAM10000

El entrenamiento supervisado usa HAM10000, un dataset publico de imagenes dermatoscopicas con siete diagnosticos en la columna `dx`:

- `akiec`: queratosis actinica / enfermedad de Bowen.
- `bcc`: carcinoma basocelular.
- `bkl`: lesiones benignas tipo queratosis.
- `df`: dermatofibroma.
- `mel`: melanoma.
- `nv`: nevus melanociticos.
- `vasc`: lesiones vasculares.

Estructura esperada:

```text
data/raw/HAM10000/
|-- HAM10000_metadata.csv
|-- HAM10000_images_part_1/
|   `-- ISIC_0024306.jpg
`-- HAM10000_images_part_2/
    `-- ISIC_0034320.jpg
```

El script busca las imagenes de forma recursiva y las une con la metadata usando `image_id`.

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

Para HAM10000 se usa otro flujo:

1. Lectura de `HAM10000_metadata.csv`.
2. Division estratificada en entrenamiento, validacion y prueba.
3. Aumento simple de datos en entrenamiento: volteos, rotacion leve, brillo y contraste.
4. CNN local `SmallDermCnn`, definida en el proyecto.
5. Inicializacion aleatoria de pesos.
6. Entrenamiento con `CrossEntropyLoss` ponderada por desbalance de clases.
7. Evaluacion con accuracy, macro F1, matriz de confusion y reporte por clase.

## Estructura

```text
.
|-- app.py                         # Interfaz web con Streamlit
|-- train.py                       # Entrenamiento desde consola
|-- train_ham10000.py              # CNN propia entrenada desde cero con HAM10000
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
|       |-- ham10000.py            # Dataset, CNN y entrenamiento supervisado
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

### Entrenamiento supervisado con HAM10000

Coloca HAM10000 en `data/raw/HAM10000/` y ejecuta:

```bash
python train_ham10000.py --data-dir data/raw/HAM10000 --metadata data/raw/HAM10000/HAM10000_metadata.csv --epochs 20 --batch-size 32
```

Si tienes GPU con CUDA:

```bash
python train_ham10000.py --device cuda --epochs 30 --batch-size 64
```

Al finalizar se generan:

- `models/ham10000_cnn_from_scratch.pt`
- `reports/ham10000_training_history.csv`
- `reports/ham10000_confusion_matrix.csv`
- `reports/ham10000_test_report.json`

La CNN usada es `SmallDermCnn`. No llama a `torchvision.models`, no carga backbones y no descarga pesos.

### Agrupamiento no supervisado

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

Para la parte supervisada del proyecto se debe usar HAM10000 en `data/raw/HAM10000/`. Para la parte exploratoria no supervisada se puede trabajar con imagenes del grupo o una subcarpeta de HAM10000.

Si se quiere ejecutar clustering con HAM10000 sin etiquetas, se puede apuntar a la carpeta raiz:

```text
data/raw/HAM10000/
```

Recomendacion para organizar la informacion:

- Usar imagenes con buena iluminacion y enfoque.
- Evitar fotos duplicadas o borrosas.
- Mantener un criterio similar de captura para reducir ruido.
- Registrar en una hoja externa el origen de las imagenes y observaciones relevantes.

El entregable supervisado debe aclarar que HAM10000 es un dataset publico usado localmente y que el entrenamiento se hizo desde cero, sin modelos preentrenados.

## Alcance actual

El proyecto implementa un prototipo funcional de analisis no supervisado y una CNN supervisada para HAM10000. No debe presentarse como diagnostico clinico; sus resultados son academicos y dependen de la particion, el balance de clases y las epocas entrenadas.

## Trabajo futuro

- Probar arquitecturas CNN propias mas profundas si se cuenta con GPU.
- Agregar UMAP o t-SNE para visualizaciones alternativas.
- Comparar resultados con etiquetas clinicas si el dataset las incluye.
- Guardar reportes graficos en PDF.
- Mejorar segmentacion con modelos especializados para lesiones.
