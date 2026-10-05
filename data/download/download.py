"""[Member 1 - Ngô Xuân Hoàng] Tải và chuẩn bị tập dữ liệu ACDC Rain từ Kaggle.

Nguồn dữ liệu:
  Kaggle Dataset: https://www.kaggle.com/datasets/njayadithya/rgb-anon-trainvaltest
  Bao gồm:
    - Ảnh trời quang tham chiếu: *_ref.png
    - Ảnh mưa thực tế: *.png trong các split train, val, test
"""
import os
import sys
from pathlib import Path


def download_acdc_rain(dest_dir: str = "dataset/raw"):
    dest = Path(dest_dir)
    dest.mkdir(parents=True, exist_ok=True)
    print(f"Downloading ACDC rain rgb_anon dataset to {dest}...")
    print("Command line via Kaggle API: kaggle datasets download -d njayadithya/rgb-anon-trainvaltest -p " + str(dest))
    print("Or on Kaggle notebook environment: available automatically at /kaggle/input/*/rgb_anon/rain")


def main():
    print("=== ACDC Rain Downloader (Member 1 - Ngô Xuân Hoàng) ===")
    download_acdc_rain()


if __name__ == "__main__":
    main()
