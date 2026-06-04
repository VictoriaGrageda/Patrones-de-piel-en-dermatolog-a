from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .clustering import ClusterResult, cluster_features
from .config import MODELS_DIR, REPORTS_DIR, SUPPORTED_EXTENSIONS, PipelineConfig
from .features import FeatureVector, extract_features
from .preprocessing import apply_mask, correct_contrast, load_image, segment_skin_region


def clustering_artifact_path(method: str) -> Path:
    return MODELS_DIR / f"skin_pattern_model_{method.lower()}.joblib"


def clustering_report_path(method: str) -> Path:
    return REPORTS_DIR / f"clustering_results_{method.lower()}.csv"


def discover_images(input_dir: str | Path) -> list[Path]:
    # Recorre la carpeta del dataset y encuentra solo archivos de imagen soportados.
    directory = Path(input_dir)
    return sorted(
        path
        for path in directory.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    )


def process_image(path: Path, config: PipelineConfig) -> FeatureVector:
    # Pipeline de vision por computadora para clustering:
    # carga imagen, mejora contraste, segmenta la region de piel y extrae rasgos.
    image = load_image(path, config.image_size)
    enhanced = correct_contrast(image)
    mask = segment_skin_region(enhanced)
    masked = apply_mask(enhanced, mask)
    return extract_features(path.name, masked, mask)


def build_feature_matrix(paths: list[Path], config: PipelineConfig) -> tuple[list[str], np.ndarray]:
    # Convierte muchas imagenes en una matriz numerica.
    # Cada fila representa una imagen y cada columna una caracteristica visual.
    if not paths:
        raise ValueError("No se encontraron imagenes en la carpeta indicada.")

    vectors = [process_image(path, config) for path in paths]
    names = [vector.name for vector in vectors]
    matrix = np.vstack([vector.values for vector in vectors])
    return names, matrix


def run_pipeline(
    input_dir: str | Path,
    method: str = "kmeans",
    config: PipelineConfig | None = None,
    save_artifacts: bool = True,
) -> tuple[pd.DataFrame, ClusterResult]:
    # Entrenamiento no supervisado:
    # no usa etiquetas del CSV; agrupa imagenes por similitud de color, textura y forma.
    config = config or PipelineConfig()
    paths = discover_images(input_dir)
    names, matrix = build_feature_matrix(paths, config)

    result = cluster_features(
        matrix,
        method=method,
        clusters=config.clusters,
        pca_components=config.pca_components,
        random_state=config.random_state,
    )

    output = pd.DataFrame(
        {
            "image": names,
            "cluster": result.labels,
            "pc1": result.embedding[:, 0],
            "pc2": result.embedding[:, 1] if result.embedding.shape[1] > 1 else 0.0,
        }
    )

    for metric, value in result.metrics.items():
        output.attrs[metric] = value

    if save_artifacts:
        # Guarda el modelo completo para que la app pueda usarlo despues con imagenes nuevas.
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        method = method.lower()
        artifact = {
            "config": config,
            "method": method,
            "cluster_model": result.model,
            "scaler": result.scaler,
            "reducer": result.reducer,
            "metrics": result.metrics,
        }

        output.to_csv(clustering_report_path(method), index=False)
        joblib.dump(artifact, clustering_artifact_path(method))

        # Mantiene compatibilidad con la version anterior de la app.
        output.to_csv(REPORTS_DIR / "clustering_results.csv", index=False)
        joblib.dump(artifact, MODELS_DIR / "skin_pattern_model.joblib")

    return output, result
