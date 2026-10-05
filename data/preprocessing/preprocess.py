"""[Member 1 - Ngô Xuân Hoàng] Chuẩn hoá kích thước và định dạng ảnh.

Quy chuẩn:
  - Độ phân giải: 1280 x 720 (tỷ lệ 16:9 chuẩn ADAS)
  - Phương pháp nội suy: cv2.INTER_AREA (giảm thiểu aliasing artifacts khi downscale từ 1920x1080)
  - Định dạng màu: RGB / BGR uint8
"""
import sys
from pathlib import Path

try:
    import cv2
except ImportError:
    cv2 = None


def preprocess_image(img, target_width: int = 1280, target_height: int = 720):
    if cv2 is None:
        raise ImportError("OpenCV (cv2) is required for image preprocessing.")
    return cv2.resize(img, (target_width, target_height), interpolation=cv2.INTER_AREA)


def main():
    print("=== Image Preprocessing Module (Member 1 - Ngô Xuân Hoàng) ===")
    print("Target resolution: 1280x720, Interpolation: INTER_AREA")
    if cv2 is None:
        print("Note: OpenCV not installed in active environment. Preprocessing runs on Kaggle GPU pipeline.")
    else:
        print("OpenCV is ready.")


if __name__ == "__main__":
    main()
