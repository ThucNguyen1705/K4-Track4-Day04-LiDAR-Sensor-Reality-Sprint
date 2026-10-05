"""[Member 2] Mức độ chi tiết."""
import cv2


def edge_density(gray, low=100, high=200):
    """Tỉ lệ pixel là cạnh Canny."""
    return float((cv2.Canny(gray, low, high) > 0).mean())
