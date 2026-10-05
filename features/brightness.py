"""[Member 2] Độ sáng / phơi sáng / tương phản. Các giá trị chuẩn hoá về [0, 1]."""
import numpy as np


def mean_brightness(gray):
    return float(gray.mean() / 255)


def brightness_std(gray):
    return float(gray.std() / 255)


def contrast(gray):
    """Khoảng tương phản p95 - p5 (bền với vài pixel cực trị hơn max - min)."""
    cdf = np.cumsum(np.bincount(gray.ravel(), minlength=256)) / gray.size
    return float((np.searchsorted(cdf, 0.95) - np.searchsorted(cdf, 0.05)) / 255)


def saturation_ratio(gray, thr=250):
    """Tỉ lệ pixel cháy sáng (overexposure / glare)."""
    return float((gray >= thr).mean())


def dark_pixel_ratio(gray, thr=20):
    """Tỉ lệ pixel quá tối (underexposure)."""
    return float((gray <= thr).mean())
