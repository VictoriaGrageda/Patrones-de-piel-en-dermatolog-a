import argparse
from pathlib import Path

from .config import PipelineConfig
from .pipeline import run_pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analisis no supervisado de patrones de piel en imagenes dermatologicas."
    )
    parser.add_argument("--input", default="data/raw", help="Carpeta con imagenes dermatologicas.")
    parser.add_argument(
        "--method",
        default="kmeans",
        choices=["kmeans", "dbscan"],
        help="Algoritmo de clustering.",
    )
    parser.add_argument("--clusters", type=int, default=4, help="Numero de clusters para K-Means, GMM o Fuzzy.")
    parser.add_argument("--pca-components", type=int, default=12, help="Componentes PCA para reducir dimensionalidad.")
    parser.add_argument("--image-size", type=int, default=224, help="Tamano cuadrado de entrada en pixeles.")
    parser.add_argument("--no-save", action="store_true", help="No guardar resultados ni modelo.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = PipelineConfig(
        image_size=(args.image_size, args.image_size),
        clusters=args.clusters,
        pca_components=args.pca_components,
    )

    output, result = run_pipeline(
        input_dir=Path(args.input),
        method=args.method,
        config=config,
        save_artifacts=not args.no_save,
    )

    print(output.to_string(index=False))
    print("\nMetricas:")
    for name, value in result.metrics.items():
        formatted = "N/A" if value is None else f"{value:.4f}"
        print(f"- {name}: {formatted}")


if __name__ == "__main__":
    main()
