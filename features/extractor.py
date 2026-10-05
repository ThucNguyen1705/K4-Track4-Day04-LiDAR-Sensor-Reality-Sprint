"""[Member 2] Trích toàn bộ IQA features cho 1 ảnh.

    from features.extractor import extract
    feats = extract(cv2.imread(path))          # dict: 11 global (+ 36 grid)
"""
import cv2

from features.blur import gradient_magnitude, laplacian_variance
from features.brightness import (brightness_std, contrast, dark_pixel_ratio, mean_brightness,
                                 saturation_ratio)
from features.color import color_statistics
from features.edges import edge_density
from features.entropy import entropy
from features.noise import noise_estimate
from features.spatial import grid_features

GRAY_FEATURES = {
    "laplacian_variance": laplacian_variance,
    "mean_brightness": mean_brightness,
    "brightness_std": brightness_std,
    "contrast": contrast,
    "entropy": entropy,
    "saturation_ratio": saturation_ratio,
    "dark_pixel_ratio": dark_pixel_ratio,
    "noise_estimate": noise_estimate,
    "edge_density": edge_density,
    "gradient_magnitude": gradient_magnitude,
}
FEATURES = list(GRAY_FEATURES) + ["color_statistics"]          # 11 global, đúng thứ tự schema
GRID_FUNCS = {k: GRAY_FEATURES[k] for k in
              ("laplacian_variance", "mean_brightness", "saturation_ratio", "edge_density")}
GRID_FEATURES = [f"{k}_r{i}c{j}" for i in range(3) for j in range(3) for k in GRID_FUNCS]


def extract(bgr, size=(1280, 720), grid=True):
    """bgr: ảnh uint8 BGR (cv2.imread). Resize về `size` để feature so sánh được giữa các nguồn."""
    if size and (bgr.shape[1], bgr.shape[0]) != tuple(size):
        bgr = cv2.resize(bgr, tuple(size), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    out = {k: fn(gray) for k, fn in GRAY_FEATURES.items()}
    out["color_statistics"] = color_statistics(bgr)
    if grid:
        out.update(grid_features(gray, GRID_FUNCS))
    return out
