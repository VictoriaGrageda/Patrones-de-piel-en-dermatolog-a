from dataclasses import dataclass

import numpy as np
from sklearn.cluster import DBSCAN, KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score
from sklearn.preprocessing import StandardScaler


@dataclass
class ClusterResult:
    # Resultado completo del clustering: etiquetas, coordenadas PCA,
    # metricas, modelo entrenado y transformadores usados.
    labels: np.ndarray
    embedding: np.ndarray
    metrics: dict[str, float | None]
    model: object
    scaler: StandardScaler
    reducer: PCA


def reduce_features(features: np.ndarray, n_components: int, random_state: int) -> tuple[np.ndarray, StandardScaler, PCA]:
    # Estandariza las caracteristicas y reduce dimensiones con PCA.
    # Las dos primeras componentes se usan como coordenadas X/Y del grafico.
    scaler = StandardScaler()
    scaled = scaler.fit_transform(features)
    max_components = max(1, min(n_components, scaled.shape[0], scaled.shape[1]))
    reducer = PCA(n_components=max_components, random_state=random_state)
    return reducer.fit_transform(scaled), scaler, reducer


def evaluate_clusters(embedding: np.ndarray, labels: np.ndarray) -> dict[str, float | None]:
    # Calcula metricas internas de clustering. No necesita etiquetas reales,
    # por eso sirve para aprendizaje no supervisado.
    unique_labels = set(labels.tolist())
    if len(unique_labels) < 2 or len(unique_labels) >= len(labels):
        return {
            "silhouette": None,
            "davies_bouldin": None,
            "calinski_harabasz": None,
        }

    return {
        "silhouette": float(silhouette_score(embedding, labels)),
        "davies_bouldin": float(davies_bouldin_score(embedding, labels)),
        "calinski_harabasz": float(calinski_harabasz_score(embedding, labels)),
    }


def cluster_features(
    features: np.ndarray,
    method: str,
    clusters: int,
    pca_components: int,
    random_state: int,
) -> ClusterResult:
    # Selecciona y entrena el algoritmo no supervisado indicado.
    # El resultado son clusters numericos, no diagnosticos medicos.
    embedding, scaler, reducer = reduce_features(features, pca_components, random_state)
    method = method.lower()
    effective_clusters = max(1, min(clusters, embedding.shape[0]))

    if method == "kmeans":
        model = KMeans(n_clusters=effective_clusters, n_init="auto", random_state=random_state)
        labels = model.fit_predict(embedding)
    elif method == "dbscan":
        model = DBSCAN(eps=1.2, min_samples=3)
        labels = model.fit_predict(embedding)
    else:
        raise ValueError(f"Metodo de clustering no soportado: {method}")

    return ClusterResult(
        labels=np.asarray(labels),
        embedding=embedding,
        metrics=evaluate_clusters(embedding, np.asarray(labels)),
        model=model,
        scaler=scaler,
        reducer=reducer,
    )
