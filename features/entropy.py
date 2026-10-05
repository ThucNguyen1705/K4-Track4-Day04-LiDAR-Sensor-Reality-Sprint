"""[Member 2] Lượng thông tin của ảnh"""
import numpy as np


def entropy(gray):
    """Shannon entropy (bit) của histogram 256 mức xám, tối đa 8"""
    p = np.bincount(gray.ravel(), minlength=256) / gray.size
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())
