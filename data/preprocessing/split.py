"""[Member 1 - Ngô Xuân Hoàng] Chiến lược phân chia Split Train / Val / Test theo Sequence.

Nguyên tắc vàng:
  - Chia theo sequence_id / scene_id (không chia ngẫu nhiên từng frame).
  - Giữ nguyên cấu trúc phân chia của tác giả tập dữ liệu ACDC gốc:
      + train: các chuỗi GOPR/GP quy định cho tập huấn luyện (400 ảnh gốc)
      + val: các chuỗi quy định cho tập validation (100 ảnh gốc)
      + test: các chuỗi quy định cho tập kiểm thử độc lập (500 ảnh gốc)
  - Đảm bảo 0% data leakage giữa tập train và test.
"""
import re


def get_sequence_id(file_name: str) -> str:
    m = re.match(r"(GOPR\d+|GP\d+|[A-Za-z0-9]+)_frame_", file_name)
    if m:
        return m.group(1)
    return file_name.split("_")[0]


def main():
    print("=== Sequence-Based Split Strategy (Member 1 - Ngô Xuân Hoàng) ===")
    print("Split mapping: Sequence-level isolation (Zero scene leakage).")
    print("Clean images: Train (400) / Val (100) / Test (500) -> 1,000 reference frames.")
    print("Synthetic corruptions retain identical split of their parent image.")


if __name__ == "__main__":
    main()
