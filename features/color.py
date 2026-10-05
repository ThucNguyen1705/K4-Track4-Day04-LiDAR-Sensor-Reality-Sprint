"""[Member 2] Thống kê màu."""
import cv2
import numpy as np


def color_statistics(bgr):
    """Colorfulness Hasler & Suesstrunk (2003): ~0 = xám/bạc màu, > 50 = màu sặc sỡ."""
    b, g, r = [c.astype(np.float32) for c in cv2.split(bgr)]
    rg, yb = r - g, 0.5 * (r + g) - b
    return float(np.hypot(rg.std(), yb.std()) + 0.3 * np.hypot(rg.mean(), yb.mean()))
