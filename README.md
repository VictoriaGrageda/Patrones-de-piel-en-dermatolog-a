# Patrones de piel en dermatologia

Proyecto de Inteligencia Artificial 1 orientado al analisis de imagenes dermatologicas mediante vision por computadora. El sistema tiene dos modulos principales:

- **Clustering exploratorio:** aprendizaje no supervisado para agrupar imagenes por similitud visual.
- **Identificacion HAM10000:** aprendizaje supervisado para clasificar imagenes en las 7 clases del dataset HAM10000 mediante una CNN propia entrenada desde cero.

> El modelo HAM10000 se entrena desde cero con las imagenes locales del dataset. No se usan modelos preentrenados, backbones externos ni pesos descargados.

## Objetivo

Construir un prototipo capaz de:

- Recibir imagenes dermatologicas.
- Preprocesar y normalizar las imagenes.
- Segmentar una region de interes aproximada.
- Extraer caracteristicas visuales interpretables.
- Entrenar un modelo de agrupamiento sin usar etiquetas.
- Entrenar una CNN propia para identificar las 7 clases de HAM10000 desde inicializacion aleatoria.
- Reducir dimensionalidad con PCA.
- Agrupar patrones con K-Means o DBSCAN.
- Evaluar la calidad de agrupamiento con metricas no supervisadas.
- Evaluar clasificacion supervisada con accuracy, macro F1, matriz de confusion y reporte por clase.
- Visualizar clusters y resultados.

## Area y subarea

- **Area:** Aprendizaje no supervisado y aprendizaje supervisado dentro de Inteligencia Artificial.
- **Subarea:** Vision por computadora aplicada al analisis de imagenes dermatologicas.

En el proyecto estas areas se aplican asi:

- **Aprendizaje no supervisado:** se usa en el modulo de clustering. El sistema no recibe diagnosticos ni etiquetas; extrae color, textura y forma de cada imagen, reduce dimensionalidad con PCA y agrupa los casos parecidos con K-Means o DBSCAN. El resultado son grupos numericos, no enfermedades.
- **Vision por computadora:** se aplica antes de ambos modulos. Permite cargar la imagen, corregir contraste, segmentar una region de interes y convertir la imagen en informacion numerica.
- **Identificacion supervisada:** se usa en el modulo HAM10000. En este caso si existen etiquetas reales (`dx`) en la metadata, por eso la CNN aprende a clasificar entre `akiec`, `bcc`, `bkl`, `df`, `mel`, `nv` y `vasc`.

## Dataset HAM10000

El entrenamiento supervisado usa HAM10000, un dataset publico de imagenes dermatoscopicas con siete diagnosticos en la columna `dx`.

Fuente del dataset usado en el proyecto:

```text
https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000?resource=download
```

Despues de descargarlo desde Kaggle, los archivos deben organizarse localmente dentro de `data/raw/HAM10000/`.

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
7. **Clustering:** K-Means o DBSCAN.
8. **Evaluacion:** Silhouette Score, Davies-Bouldin Index y Calinski-Harabasz Index.
9. **Interpretacion:** tabla de imagenes agrupadas y mapa PCA en 2D.

### Modulo 2: identificacion supervisada HAM10000

Para HAM10000 se usa otro flujo:

1. Lectura de `HAM10000_metadata.csv`.
2. Division estratificada en entrenamiento, validacion y prueba.
3. Aumento simple de datos en entrenamiento: volteos, rotacion leve, brillo y contraste.
4. CNN local `SmallDermCnn`, definida en el proyecto.
5. Inicializacion aleatoria de pesos.
6. Entrenamiento con `CrossEntropyLoss` ponderada por desbalance de clases.
7. Evaluacion con accuracy, macro F1, matriz de confusion y reporte por clase.

Este modulo si realiza identificacion/clasificacion porque aprende con etiquetas reales del dataset. Aun asi, sus salidas son resultados academicos y no diagnosticos clinicos.

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

## Como ejecutar el proyecto desde cero

Sigue estos pasos desde una terminal ubicada en la carpeta donde quieres tener el proyecto.

### 1. Obtener el proyecto

Si todavia no tienes el repositorio clonado:

```bash
git clone <URL_DEL_REPOSITORIO>
cd Patrones-de-piel-en-dermatolog-a
```

Si ya tienes la carpeta del proyecto, solo entra a ella:

```bash
cd Patrones-de-piel-en-dermatolog-a
```

### 2. Verificar requisitos

Necesitas:

- Python 3.10 o superior.
- pip.
- Git, solo si vas a clonar el repositorio.
- Imagenes dermatologicas para probar el clustering o el dataset HAM10000 para el entrenamiento supervisado.

Puedes verificar Python con:

```bash
python --version
pip --version
```

### 3. Crear y activar el entorno virtual

En Windows PowerShell:

```bash
python -m venv .venv
.\.venv\Scripts\activate
```

En macOS o Linux:

```bash
python -m venv .venv
source .venv/bin/activate
```

Cuando el entorno este activo, deberias ver `(.venv)` al inicio de la terminal.

### 4. Instalar dependencias

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Instalacion editable opcional, util si quieres ejecutar los comandos declarados en `pyproject.toml`:

```bash
pip install -e .
```

### 5. Preparar datos de entrada

Para probar el flujo no supervisado, coloca imagenes JPG, PNG, BMP o WEBP dentro de:

```text
data/raw/
```

Si quieres entrenar con HAM10000, descarga el dataset desde Kaggle:

```text
https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000?resource=download
```

Luego descomprime los archivos y deja la estructura asi:

```text
data/raw/HAM10000/
|-- HAM10000_metadata.csv
|-- HAM10000_images_part_1/
|   `-- ISIC_0024306.jpg
`-- HAM10000_images_part_2/
    `-- ISIC_0034320.jpg
```

### 6. Ejecutar el proyecto

Para abrir la interfaz web:

```bash
streamlit run app.py
```

Luego abre en el navegador la URL que muestre Streamlit, normalmente:

```text
http://localhost:8501
```

Desde la interfaz puedes cargar imagenes, elegir el algoritmo de agrupamiento, entrenar el modelo seleccionado y visualizar los resultados. En la pestana **Clustering exploratorio** se puede escoger entre:

- K-Means.
- DBSCAN.

K-Means usa el valor **Numero de clusters**. DBSCAN no usa ese valor porque forma grupos por densidad y tambien puede marcar imagenes como ruido o sin cluster.

Para usar la prediccion HAM10000 en la app, primero debe existir el modelo entrenado en:

```text
models/ham10000_cnn_from_scratch.pt
```

Para ejecutar clustering desde consola con K-Means:

```bash
python train.py --input data/raw --method kmeans --clusters 4
```

Otro metodo disponible para clustering:

```bash
python train.py --input data/raw --method dbscan
```

Cada entrenamiento guarda un modelo y un reporte separados por metodo. Asi puedes entrenar varios algoritmos sin perder el anterior:

```text
models/skin_pattern_model_kmeans.joblib
models/skin_pattern_model_dbscan.joblib
reports/clustering_results_kmeans.csv
reports/clustering_results_dbscan.csv
```

Tambien se actualizan estos archivos generales para compatibilidad con versiones anteriores de la app:

```text
models/skin_pattern_model.joblib
reports/clustering_results.csv
```

Si usaste las imagenes sinteticas de prueba:

```bash
python train.py --input data/raw --method kmeans --clusters 3
```

Para entrenar la CNN con HAM10000:

```bash
python train_ham10000.py --data-dir data/raw/HAM10000 --metadata data/raw/HAM10000/HAM10000_metadata.csv --epochs 20 --batch-size 32
```

Entrenamiento rapido de prueba con una muestra reducida:

```bash
python train_ham10000.py --data-dir data/raw/HAM10000 --metadata data/raw/HAM10000/HAM10000_metadata.csv --epochs 1 --batch-size 32 --image-size 64 --max-samples 1000
```

Si tienes GPU con CUDA:

```bash
python train_ham10000.py --data-dir data/raw/HAM10000 --metadata data/raw/HAM10000/HAM10000_metadata.csv --device cuda --epochs 30 --batch-size 64
```

### 7. Revisar resultados generados

Despues de entrenar o ejecutar clustering, revisa:

```text
reports/
models/
```

Los archivos principales son:

- `reports/clustering_results.csv`
- `models/skin_pattern_model.joblib`
- `reports/clustering_results_<metodo>.csv`
- `models/skin_pattern_model_<metodo>.joblib`
- `models/ham10000_cnn_from_scratch.pt`
- `reports/ham10000_training_history.csv`
- `reports/ham10000_confusion_matrix.csv`
- `reports/ham10000_test_report.json`

## Como se aplican los dos sectores del proyecto

### 1. Sector de clustering

Este sector corresponde al aprendizaje no supervisado. Su objetivo es encontrar grupos de imagenes con patrones visuales similares sin usar etiquetas medicas.

Flujo:

1. Lee imagenes desde `data/raw/` o desde una subcarpeta como `data/raw/HAM10000/`.
2. Aplica vision por computadora: carga, redimensionamiento, contraste y segmentacion aproximada.
3. Extrae caracteristicas de color, textura y forma.
4. Estandariza las caracteristicas y aplica PCA.
5. Agrupa con K-Means o DBSCAN.
6. Guarda el reporte y modelo por metodo, por ejemplo `reports/clustering_results_kmeans.csv` y `models/skin_pattern_model_kmeans.joblib`.

Interpretacion: un cluster indica similitud visual entre imagenes. No equivale automaticamente a una enfermedad.

### 2. Sector de identificacion

Este sector corresponde al aprendizaje supervisado. Usa HAM10000 porque ese dataset trae etiquetas reales en `HAM10000_metadata.csv`.

Flujo:

1. Une cada `image_id` del CSV con su archivo de imagen.
2. Divide los datos en entrenamiento, validacion y prueba.
3. Entrena la CNN `SmallDermCnn` desde cero.
4. Evalua con metricas de clasificacion.
5. Guarda `models/ham10000_cnn_from_scratch.pt` y reportes en `reports/`.

Interpretacion: el modelo devuelve probabilidades para las 7 clases HAM10000. Es una identificacion academica basada en el dataset, no una decision medica final.

## Dataset del proyecto

Para la parte supervisada del proyecto se debe usar HAM10000 en `data/raw/HAM10000/`. El dataset usado se obtuvo desde Kaggle:

```text
https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000?resource=download
```

Para la parte exploratoria no supervisada se puede trabajar con imagenes del grupo o una subcarpeta de HAM10000.

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
