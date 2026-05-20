from dataclasses import dataclass

import numpy as np
from skimage import color, feature, measure


@dataclass(frozen=True)
class FeatureVector:
    name: str
    values: np.ndarray


def _safe_stats(values: np.ndarray) -> list[float]:
    if values.size == 0:
        return [0.0, 0.0, 0.0, 0.0]
    return [
        float(values.mean()),
        float(values.std()),
        float(np.percentile(values, 25)),
        float(np.percentile(values, 75)),
    ]


def color_features(image: np.ndarray, mask: np.ndarray) -> list[float]:
    hsv = color.rgb2hsv(image)
    lab = color.rgb2lab(image)
    selected_rgb = image[mask]
    selected_hsv = hsv[mask]
    selected_lab = lab[mask]

    values: list[float] = []
    for channel_set in (selected_rgb, selected_hsv, selected_lab):
        for channel in range(channel_set.shape[1]):
            values.extend(_safe_stats(channel_set[:, channel]))
    return values


def texture_features(image: np.ndarray, mask: np.ndarray) -> list[float]:
    gray = color.rgb2gray(image)
    gray_uint = np.clip(gray * 255, 0, 255).astype(np.uint8)

    lbp = feature.local_binary_pattern(gray_uint, P=16, R=2, method="uniform")
    lbp_values = lbp[mask]
    hist, _ = np.histogram(lbp_values, bins=18, range=(0, 18), density=True)

    hog = feature.hog(
        gray,
        orientations=8,
        pixels_per_cell=(32, 32),
        cells_per_block=(1, 1),
        feature_vector=True,
    )
    hog_summary = _safe_stats(hog)
    return [float(v) for v in hist] + hog_summary


def shape_features(mask: np.ndarray) -> list[float]:
    labeled = measure.label(mask)
    regions = measure.regionprops(labeled)
    if not regions:
        return [0.0] * 7

    region = max(regions, key=lambda item: item.area)
    area = float(region.area)
    bbox_area = float(region.bbox_area) or 1.0
    perimeter = float(region.perimeter) or 1.0
    circularity = float((4 * np.pi * area) / (perimeter**2))

    return [
        area / mask.size,
        perimeter / mask.size,
        float(region.eccentricity),
        float(region.solidity),
        float(region.extent),
        area / bbox_area,
        circularity,
    ]


def extract_features(name: str, image: np.ndarray, mask: np.ndarray) -> FeatureVector:
    values = color_features(image, mask) + texture_features(image, mask) + shape_features(mask)
    return FeatureVector(name=name, values=np.asarray(values, dtype=np.float32))
