# Informe del proyecto

## Titulo

Patrones de piel en dermatologia aplicando aprendizaje no supervisado y vision por computadora.

## Planteamiento del problema

El analisis de lesiones cutaneas requiere observacion especializada y comparacion de multiples rasgos visuales como color, textura, forma y bordes. En contextos academicos y de apoyo exploratorio, un sistema de inteligencia artificial puede ayudar a descubrir grupos de imagenes con caracteristicas similares sin depender inicialmente de etiquetas clinicas.

## Objetivo general

Desarrollar un prototipo de IA que procese imagenes dermatologicas y agrupe patrones de piel mediante algoritmos de aprendizaje no supervisado.

## Objetivos especificos

- Implementar un flujo de preprocesamiento para normalizar imagenes.
- Extraer caracteristicas visuales de color, textura y forma.
- Entrenar los modelos con imagenes propias del proyecto, sin pesos preentrenados.
- Aplicar reduccion de dimensionalidad con PCA.
- Comparar algoritmos de clustering: K-Means, GMM, DBSCAN y Fuzzy C-Means.
- Evaluar los agrupamientos con metricas no supervisadas.
- Visualizar los resultados en una interfaz simple.

## Justificacion

El aprendizaje no supervisado es adecuado cuando no se dispone de etiquetas clinicas confiables o cuando se desea explorar la estructura interna de un conjunto de imagenes. En dermatologia, este enfoque permite identificar patrones visuales recurrentes y posibles casos atipicos que podrian requerir revision posterior.

## Marco teorico resumido

### Vision por computadora

Permite transformar imagenes en datos numericos analizables. En este proyecto se usan rasgos de color, textura y forma para representar cada imagen como un vector de caracteristicas.

### Aprendizaje no supervisado

Agrupa datos sin utilizar etiquetas de clase. El sistema busca similitudes entre imagenes y asigna cada muestra a un cluster.

En este proyecto el entrenamiento se realiza desde cero con las imagenes cargadas en el repositorio. No se utilizan redes neuronales preentrenadas, pesos externos ni servicios de IA externos.

### Clustering difuso

Fuzzy C-Means permite que una imagen tenga grados de pertenencia a varios clusters. Esto es util en dominios medicos donde los limites entre patrones pueden ser ambiguos.

### PCA

Principal Component Analysis reduce la dimensionalidad de las caracteristicas y conserva la mayor variacion posible. Facilita el clustering y la visualizacion.

## Flujo implementado

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

- Agrupamiento de imagenes con patrones visuales similares.
- Mapa PCA con separacion aproximada entre clusters.
- Tabla con imagen, cluster asignado y coordenadas principales.
- Metricas de calidad para comparar algoritmos.

## Datos de entrenamiento

El conjunto de entrenamiento corresponde a las imagenes reunidas por el grupo y ubicadas en `data/raw/`. Las imagenes sinteticas del script de prueba solo sirven para verificar que el programa ejecuta correctamente; no representan el resultado final del proyecto.

## Limitaciones

- No realiza diagnostico clinico.
- La segmentacion es aproximada y puede fallar con fondos complejos.
- El desempeno depende de la calidad y cantidad de imagenes.
- Para clasificacion medica real se requieren etiquetas, validacion experta y control de sesgos.
- Si se desea entrenar una red neuronal desde cero, se requiere un dataset mucho mas grande que el usado para una demostracion academica.

## Conclusion

El prototipo demuestra como aplicar tecnicas de IA 1, vision por computadora y aprendizaje no supervisado para explorar patrones dermatologicos. La arquitectura queda preparada para incorporar mas imagenes propias, comparar algoritmos y mejorar la evaluacion sin depender de modelos preentrenados.
