from dataclasses import dataclass
from pathlib import Path


SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass(frozen=True)
class PipelineConfig:
    image_size: tuple[int, int] = (224, 224)
    clusters: int = 4
    pca_components: int = 12
    random_state: int = 42
    fuzzy_m: float = 2.0
    fuzzy_max_iter: int = 150
    fuzzy_error: float = 1e-5


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
