# Informe del proyecto

## Titulo

Patrones de piel en dermatologia aplicando vision por computadora y entrenamiento propio con HAM10000.

## Planteamiento del problema

El analisis de lesiones cutaneas requiere observacion especializada y comparacion de multiples rasgos visuales como color, textura, forma y bordes. En contextos academicos, un sistema de inteligencia artificial puede explorar similitudes visuales y tambien aprender a clasificar lesiones cuando existe un dataset etiquetado como HAM10000.

## Objetivo general

Desarrollar un prototipo de IA que procese imagenes dermatologicas y entrene un modelo propio, sin pesos preentrenados, usando el dataset HAM10000.

## Objetivos especificos

- Implementar un flujo de preprocesamiento para normalizar imagenes.
- Preparar HAM10000 a partir de `HAM10000_metadata.csv` y sus carpetas de imagenes.
- Entrenar una CNN propia desde cero para las siete clases `dx`.
- Evaluar clasificacion con accuracy, macro F1, matriz de confusion y reporte por clase.
- Extraer caracteristicas visuales de color, textura y forma.
- Entrenar los modelos con imagenes propias del proyecto, sin pesos preentrenados.
- Aplicar reduccion de dimensionalidad con PCA.
- Comparar algoritmos de clustering: K-Means, GMM, DBSCAN y Fuzzy C-Means.
- Evaluar los agrupamientos con metricas no supervisadas.
- Visualizar los resultados en una interfaz simple.

## Justificacion

HAM10000 permite trabajar con aprendizaje supervisado porque incluye etiquetas diagnosticas en la columna `dx`. Para cumplir el requisito academico, el proyecto no usa modelos preentrenados; define una CNN propia e inicializa sus pesos aleatoriamente. El flujo no supervisado se conserva como analisis exploratorio complementario.

## Marco teorico resumido

### Vision por computadora

Permite transformar imagenes en datos numericos analizables. En este proyecto se usan rasgos de color, textura y forma para representar cada imagen como un vector de caracteristicas.

### Aprendizaje no supervisado

Agrupa datos sin utilizar etiquetas de clase. El sistema busca similitudes entre imagenes y asigna cada muestra a un cluster.

En este proyecto el entrenamiento se realiza desde cero con las imagenes cargadas en el repositorio. No se utilizan redes neuronales preentrenadas, pesos externos ni servicios de IA externos.

### Aprendizaje supervisado con HAM10000

HAM10000 contiene siete clases: `akiec`, `bcc`, `bkl`, `df`, `mel`, `nv` y `vasc`. El entrenamiento implementado divide la metadata de forma estratificada en entrenamiento, validacion y prueba. La CNN `SmallDermCnn` se entrena con `CrossEntropyLoss` ponderada para reducir el efecto del desbalance de clases.

### Clustering difuso

Fuzzy C-Means permite que una imagen tenga grados de pertenencia a varios clusters. Esto es util en dominios medicos donde los limites entre patrones pueden ser ambiguos.

### PCA

Principal Component Analysis reduce la dimensionalidad de las caracteristicas y conserva la mayor variacion posible. Facilita el clustering y la visualizacion.

## Flujo implementado

### HAM10000 supervisado

1. Carga de `HAM10000_metadata.csv`.
2. Busqueda recursiva de imagenes en `data/raw/HAM10000/`.
3. Union por `image_id`.
4. Division estratificada en entrenamiento, validacion y prueba.
5. Aumento simple de datos en entrenamiento.
6. Entrenamiento de CNN propia desde cero.
7. Guardado del mejor modelo segun macro F1 de validacion.
8. Evaluacion final en prueba.

### Clustering exploratorio

1. Carga de imagen.
2. Correccion de orientacion y redimensionamiento.
3. Mejora de contraste.
4. Segmentacion aproximada de region de interes.
5. Extraccion de caracteristicas:
   - RGB, HSV y LAB.
   - Local Binary Patterns.
   - HOG.
   - Rasgos geometricos.
6. Normalizacion de caracteristicas.
7. Reduccion con PCA.
8. Entrenamiento del clustering con la informacion propia.
9. Evaluacion y visualizacion.

## Resultados esperados

- Modelo `models/ham10000_cnn_from_scratch.pt`.
- Historial `reports/ham10000_training_history.csv`.
- Matriz de confusion `reports/ham10000_confusion_matrix.csv`.
- Reporte de prueba `reports/ham10000_test_report.json`.
- Agrupamiento de imagenes con patrones visuales similares.
- Mapa PCA con separacion aproximada entre clusters.
- Tabla con imagen, cluster asignado y coordenadas principales.
- Metricas de calidad para comparar algoritmos.

## Datos de entrenamiento

El conjunto de entrenamiento supervisado corresponde a HAM10000 ubicado en `data/raw/HAM10000/`, con `HAM10000_metadata.csv` y las carpetas de imagenes originales. Las imagenes sinteticas del script de prueba solo sirven para verificar que el clustering ejecuta correctamente; no representan el resultado final del proyecto.

## Limitaciones

- No realiza diagnostico clinico.
- La segmentacion es aproximada y puede fallar con fondos complejos.
- El desempeno depende de la calidad y cantidad de imagenes.
- HAM10000 esta desbalanceado; por eso se reporta macro F1 y se usa perdida ponderada.
- Para clasificacion medica real se requiere validacion experta, control de sesgos y evaluacion externa.

## Conclusion

El prototipo demuestra como aplicar vision por computadora en dermatologia con dos enfoques: clasificacion supervisada de HAM10000 mediante una CNN entrenada desde cero y clustering exploratorio con caracteristicas visuales. La solucion cumple la restriccion de no depender de modelos preentrenados.
