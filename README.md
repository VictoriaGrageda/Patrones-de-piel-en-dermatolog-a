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

Si quieres entrenar con HAM10000, la estructura debe quedar asi:

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

Desde la interfaz puedes cargar imagenes, elegir el algoritmo de agrupamiento y visualizar los resultados. Para usar la prediccion HAM10000 en la app, primero debe existir el modelo entrenado en:

```text
models/ham10000_cnn_from_scratch.pt
```

Para ejecutar clustering desde consola:

```bash
python train.py --input data/raw --method kmeans --clusters 4
```

Si usaste las imagenes sinteticas de prueba:

```bash
python train.py --input data/raw --method kmeans --clusters 3
```

Para entrenar la CNN con HAM10000:

```bash
python train_ham10000.py --data-dir data/raw/HAM10000 --metadata data/raw/HAM10000/HAM10000_metadata.csv --epochs 20 --batch-size 32
```

Si tienes GPU con CUDA:

```bash
python train_ham10000.py --device cuda --epochs 30 --batch-size 64
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
- `models/ham10000_cnn_from_scratch.pt`
- `reports/ham10000_training_history.csv`
- `reports/ham10000_confusion_matrix.csv`
- `reports/ham10000_test_report.json`

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
