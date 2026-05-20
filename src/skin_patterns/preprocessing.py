from pathlib import Path

import numpy as np
from PIL import Image, ImageOps
from skimage import color, exposure, filters, morphology, transform


def load_image(path: str | Path, image_size: tuple[int, int]) -> np.ndarray:
    """Load an image as normalized RGB with deterministic orientation and size."""
    image = Image.open(path)
    image = ImageOps.exif_transpose(image).convert("RGB")
    array = np.asarray(image, dtype=np.float32) / 255.0
    resized = transform.resize(
        array,
        image_size,
        anti_aliasing=True,
        preserve_range=True,
    )
    return np.clip(resized, 0.0, 1.0).astype(np.float32)


def correct_contrast(image: np.ndarray) -> np.ndarray:
    """Improve local contrast without changing the image domain."""
    return exposure.equalize_adapthist(image, clip_limit=0.02).astype(np.float32)


def segment_skin_region(image: np.ndarray) -> np.ndarray:
    """Create a lesion/skin-interest mask using color and intensity cues."""
    lab = color.rgb2lab(image)
    gray = color.rgb2gray(image)
    chroma = np.sqrt(lab[..., 1] ** 2 + lab[..., 2] ** 2)

    try:
        gray_threshold = filters.threshold_otsu(gray)
        chroma_threshold = filters.threshold_otsu(chroma)
    except ValueError:
        return np.ones(gray.shape, dtype=bool)

    mask = (gray < gray_threshold) | (chroma > chroma_threshold)
    mask = morphology.remove_small_objects(mask, min_size=64)
    mask = morphology.remove_small_holes(mask, area_threshold=64)
    mask = morphology.binary_closing(mask, morphology.disk(3))

    if mask.mean() < 0.02 or mask.mean() > 0.95:
        return np.ones(gray.shape, dtype=bool)
    return mask.astype(bool)


def apply_mask(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    masked = image.copy()
    background = image[~mask].mean(axis=0) if np.any(~mask) else np.array([1.0, 1.0, 1.0])
    masked[~mask] = background
    return masked.astype(np.float32)
