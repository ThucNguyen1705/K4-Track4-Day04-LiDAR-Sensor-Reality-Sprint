"""[Member 3] Sản phẩm cuối: ảnh camera -> 11 IQA features (Member 2) -> XGBoost -> Camera Health.

    python -m health_model.predict front.png left.png right.png
    python -m health_model.predict dataset/test/ --csv outputs/results/health_scores.csv

In ra health in [0, 1], class (Healthy/Degraded/Critical) và camera weight cho fusion (= health).
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from xgboost import XGBRegressor

from common.config import load_config
from features.extractor import extract
from health_model.common import health_class, prepare_X
from health_model.train_xgboost import features_path

IMG_EXT = {".png", ".jpg", ".jpeg", ".bmp"}


class CameraHealth:
    def __init__(self, cfg=None):
        self.cfg = cfg or load_config()
        self.model = XGBRegressor()
        self.model.load_model(self.cfg["paths"]["model"])
        self.cols = json.load(open(features_path(self.cfg["paths"]["model"])))
        self.size = (self.cfg["image"]["width"], self.cfg["image"]["height"])

    def __call__(self, bgr_images):
        feats = pd.DataFrame([extract(im, self.size, grid=len(self.cols) > 11) for im in bgr_images])
        health = np.clip(self.model.predict(prepare_X(feats, self.cols)), 0, 1)
        return health, health_class(health, self.cfg)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("inputs", nargs="+", help="file ảnh hoặc thư mục")
    ap.add_argument("--csv", help="lưu kết quả ra CSV")
    args = ap.parse_args()
    paths = []
    for s in map(Path, args.inputs):
        paths += sorted(f for f in s.rglob("*") if f.suffix.lower() in IMG_EXT) if s.is_dir() else [s]

    scorer = CameraHealth()
    imgs = [cv2.imread(str(p)) for p in paths]
    health, cls = scorer(imgs)
    df = pd.DataFrame({"image": [p.name for p in paths], "health": health.round(3),
                       "class": cls, "camera_weight": health.round(3)})
    print(df.to_string(index=False))
    if args.csv:
        df.to_csv(args.csv, index=False)


if __name__ == "__main__":
    main()
