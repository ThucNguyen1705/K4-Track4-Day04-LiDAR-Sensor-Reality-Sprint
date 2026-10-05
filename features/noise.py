"""[Member 2] Ước lượng nhiễu"""
import cv2
import numpy as np

_K = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], np.float32)


def noise_estimate(gray):
    """Sigma nhiễu Gaussian theo Immerkaer (1996), đơn vị mức xám 0-255"""
    conv = cv2.filter2D(gray.astype(np.float32), -1, _K)[1:-1, 1:-1]
    return float(np.sqrt(np.pi / 2) * np.abs(conv).sum() / (6 * conv.size))
