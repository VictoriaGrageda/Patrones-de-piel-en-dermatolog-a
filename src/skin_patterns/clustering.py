from dataclasses import dataclass

import numpy as np
from sklearn.cluster import DBSCAN, KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler


@dataclass
class ClusterResult:
    labels: np.ndarray
    embedding: np.ndarray
    metrics: dict[str, float | None]
    model: object
    scaler: StandardScaler
    reducer: PCA


def reduce_features(features: np.ndarray, n_components: int, random_state: int) -> tuple[np.ndarray, StandardScaler, PCA]:
    scaler = StandardScaler()
    scaled = scaler.fit_transform(features)
    max_components = max(1, min(n_components, scaled.shape[0], scaled.shape[1]))
    reducer = PCA(n_components=max_components, random_state=random_state)
    return reducer.fit_transform(scaled), scaler, reducer


def evaluate_clusters(embedding: np.ndarray, labels: np.ndarray) -> dict[str, float | None]:
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


def fuzzy_c_means(
    data: np.ndarray,
    clusters: int,
    m: float = 2.0,
    max_iter: int = 150,
    error: float = 1e-5,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(random_state)
    membership = rng.random((clusters, data.shape[0]))
    membership = membership / membership.sum(axis=0, keepdims=True)

    for _ in range(max_iter):
        previous = membership.copy()
        weighted = membership**m
        centers = weighted @ data / weighted.sum(axis=1, keepdims=True)
        distances = np.linalg.norm(data[None, :, :] - centers[:, None, :], axis=2)
        distances = np.fmax(distances, np.finfo(np.float64).eps)

        inverse = distances ** (-2 / (m - 1))
        membership = inverse / inverse.sum(axis=0, keepdims=True)

        if np.linalg.norm(membership - previous) < error:
            break

    labels = membership.argmax(axis=0)
    return labels, centers, membership


def cluster_features(
    features: np.ndarray,
    method: str,
    clusters: int,
    pca_components: int,
    random_state: int,
    fuzzy_m: float,
    fuzzy_max_iter: int,
    fuzzy_error: float,
) -> ClusterResult:
    embedding, scaler, reducer = reduce_features(features, pca_components, random_state)
    method = method.lower()
    effective_clusters = max(1, min(clusters, embedding.shape[0]))

    if method == "kmeans":
        model = KMeans(n_clusters=effective_clusters, n_init="auto", random_state=random_state)
        labels = model.fit_predict(embedding)
    elif method == "gmm":
        model = GaussianMixture(n_components=effective_clusters, random_state=random_state)
        labels = model.fit_predict(embedding)
    elif method == "dbscan":
        model = DBSCAN(eps=1.2, min_samples=3)
        labels = model.fit_predict(embedding)
    elif method in {"fuzzy", "fcm", "fuzzy_c_means"}:
        labels, centers, membership = fuzzy_c_means(
            embedding,
            clusters=effective_clusters,
            m=fuzzy_m,
            max_iter=fuzzy_max_iter,
            error=fuzzy_error,
            random_state=random_state,
        )
        model = {"centers": centers, "membership": membership}
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
