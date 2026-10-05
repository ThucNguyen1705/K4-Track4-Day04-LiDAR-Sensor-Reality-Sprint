"""[Member 1 - Ngô Xuân Hoàng] Sinh ảnh degradation:
  - rain L1-L3: hạt mưa (streaks), sương mờ (haze), giọt nước đọng trên kính (blur)
  - blur L1-L4: Gaussian blur (sigma = 1.0, 2.0, 3.5, 6.0)
  - noise L1-L3: Gaussian sensor noise (sigma = 10, 25, 45)
  - dark L1-L2: phi tuyến gamma + suy giảm cường độ ánh sáng
  - overexposure L1-L2: cháy sáng + bão hòa cảm biến

Đảm bảo thuộc tính pixel-aligned: không làm biến đổi hình học (không dịch chuyển, không xoay),
giúp bounding box của ground truth trên ảnh sạch hoàn toàn khớp với ảnh suy biến.
"""
import argparse
import sys
from pathlib import Path

try:
    import cv2
    import numpy as np
except ImportError:
    cv2 = None
    np = None

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass


def synth_rain(img, k, rng):
    """Mô phỏng mưa 3 cấp độ: kết hợp vệt mưa ngẫu nhiên, tán xạ haze và mờ cục bộ."""
    if cv2 is None or np is None:
        raise ImportError("cv2 and numpy are required for degradation simulation.")
    H, W = img.shape[:2]
    n = [800, 2000, 4000][k - 1]
    length = [15, 25, 40][k - 1]
    alpha = [0.5, 0.65, 0.8][k - 1]
    haze = [0.08, 0.16, 0.25][k - 1]
    blur_k = [0.6, 1.0, 1.6][k - 1]

    # Lớp sương mù / haze do hơi nước trong không khí
    out = img.astype(np.float32) * (1.0 - haze) + 200.0 * haze
    out = cv2.GaussianBlur(out, (0, 0), blur_k)

    # Lớp vệt mưa (rain streaks)
    layer = np.zeros((H, W), dtype=np.float32)
    ang = np.deg2rad(rng.uniform(-15, 15))
    dx, dy = int(length * np.sin(ang)), int(length * np.cos(ang))

    xs = rng.integers(0, W, n)
    ys = rng.integers(0, H, n)
    vals = rng.uniform(0.6, 1.0, n)
    for x, y, v in zip(xs, ys, vals):
        cv2.line(layer, (int(x), int(y)), (int(x) + dx, int(y) + dy), float(v), 1)

    layer = cv2.GaussianBlur(layer, (3, 3), 0)[..., None] * alpha
    corrupted = np.clip(out * (1.0 - layer) + 230.0 * layer, 0, 255).astype(np.uint8)
    return corrupted


def synth_blur(img, k):
    """Mô phỏng mất nét cảm biến / rung lắc camera với sigma = 1.0, 2.0, 3.5, 6.0."""
    sigmas = [1.0, 2.0, 3.5, 6.0]
    return cv2.GaussianBlur(img, (0, 0), sigmas[k - 1])


def synth_noise(img, k, rng):
    """Mô phỏng nhiễu cảm biến ISO cao (Gaussian noise sigma = 10, 25, 45)."""
    sigmas = [10, 25, 45]
    noise = rng.normal(0, sigmas[k - 1], img.shape)
    return np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)


def synth_dark(img, k):
    """Mô phỏng điều kiện thiếu sáng cực độ qua phi tuyến gamma và suy giảm giá trị."""
    gammas = [1.3, 1.6]
    scales = [0.35, 0.15]
    f = (img.astype(np.float32) / 255.0) ** gammas[k - 1] * scales[k - 1]
    return np.clip(f * 255.0, 0, 255).astype(np.uint8)


def synth_overexposure(img, k):
    """Mô phỏng ánh sáng chói lóa ngược chiều / cháy sáng cảm biến."""
    scales = [1.8, 2.8]
    offsets = [0.08, 0.20]
    f = (img.astype(np.float32) / 255.0) * scales[k - 1] + offsets[k - 1]
    return np.clip(f * 255.0, 0, 255).astype(np.uint8)


def degrade(img, kind: str, level: int, rng=None):
    """Hàm tổng quát áp dụng suy biến hình ảnh."""
    if kind == "none" or level == 0:
        return img
    if rng is None:
        rng = np.random.default_rng(42)

    if kind == "rain":
        return synth_rain(img, level, rng)
    elif kind == "blur":
        return synth_blur(img, level)
    elif kind == "noise":
        return synth_noise(img, level, rng)
    elif kind == "dark":
        return synth_dark(img, level)
    elif kind == "overexposure":
        return synth_overexposure(img, level)
    else:
        raise ValueError(f"Unknown degradation type: {kind}")


def main():
    print("=== Sensor Degradation Engine (Member 1 - Ngô Xuân Hoàng) ===")
    print("Available degradations: none, rain(L1-3), blur(L1-4), noise(L1-3), dark(L1-2), overexposure(L1-2)")
    print("Pixel-aligned property: Preserves exact coordinates for zero-leakage ADAS bounding box matching.")
    if cv2 is None or np is None:
        print("Note: OpenCV/numpy not installed in current environment. Ready for runtime on Kaggle / target venv.")
    else:
        dummy = np.full((720, 1280, 3), 128, dtype=np.uint8)
        rng = np.random.default_rng(42)
        res = degrade(dummy, "blur", 2, rng)
        print(f"Test run successful: dummy shape {dummy.shape} -> corrupted shape {res.shape}")


if __name__ == "__main__":
    main()
