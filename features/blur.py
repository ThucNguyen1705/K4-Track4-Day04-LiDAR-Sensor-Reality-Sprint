"""[Member 2] Sharpness / blur."""
import cv2
import numpy as np


def laplacian_variance(gray):
    """Phương sai Laplacian: ảnh nét -> lớn, ảnh mờ -> nhỏ (noise cũng làm tăng)"""
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def gradient_magnitude(gray):
    """Độ lớn gradient Sobel trung bình"""
    g = gray.astype(np.float32)
    gx, gy = cv2.Sobel(g, cv2.CV_32F, 1, 0), cv2.Sobel(g, cv2.CV_32F, 0, 1)
    return float(np.sqrt(gx ** 2 + gy ** 2).mean())
