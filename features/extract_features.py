"""[Member 2] Đọc dataset/metadata.csv (Member 1), trích features cho mọi ảnh.

Output: outputs/features/features.csv
  image_id, parent_id, split, condition, degradation_type, degradation_level, is_synthetic,
  <11 global features>, <36 grid features>

Chạy: python -m features.extract_features [--no-grid] [--workers 4]
"""
import argparse
import os
from multiprocessing import Pool

import cv2
import pandas as pd
from tqdm import tqdm

from common.config import ROOT, load_config
from features.extractor import extract

META = ["image_id", "parent_id", "split", "condition", "degradation_type", "degradation_level",
        "is_synthetic"]


def _work(args):
    path, size, grid = args
    img = cv2.imread(str(path))
    return None if img is None else extract(img, size, grid)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--no-grid", action="store_true")
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    args = ap.parse_args()
    cfg = load_config()
    size = (cfg["image"]["width"], cfg["image"]["height"])

    meta = pd.read_csv(cfg["paths"]["metadata"])
    paths = [ROOT / p for p in meta["file_path"]]
    with Pool(args.workers) as pool:
        feats = list(tqdm(pool.imap(_work, [(p, size, not args.no_grid) for p in paths], chunksize=8),
                          total=len(paths)))
    ok = [f is not None for f in feats]
    print(f"đọc lỗi {len(ok) - sum(ok)} ảnh")
    out = pd.concat([meta.loc[ok, [c for c in META if c in meta]].reset_index(drop=True),
                     pd.DataFrame([f for f in feats if f is not None])], axis=1)
    cfg["paths"]["features"].parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(cfg["paths"]["features"], index=False)
    print(f"{len(out)} rows x {out.shape[1]} cols -> {cfg['paths']['features']}")


if __name__ == "__main__":
    main()
