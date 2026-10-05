"""[Member 3] Phần dùng chung của health_model: tên cột, path, load dữ liệu."""
import argparse
import re

import numpy as np
import pandas as pd

from common.config import ROOT, load_config
from features.extractor import FEATURES, GRID_FEATURES  # 11 global + 36 grid (Member 2)

META = ["image_id", "parent_id", "split", "condition", "degradation_type",
        "degradation_level", "is_synthetic"]
# feature có đuôi dài -> log1p (không ảnh hưởng model cây, giúp Linear Regression)
LOG_FEATURES = ["laplacian_variance", "noise_estimate", "gradient_magnitude"]
CLASSES = ["Healthy", "Degraded", "Critical"]


def get_cfg(description=""):
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--mock", action="store_true", help="dùng dữ liệu giả trong outputs/mock/")
    args = p.parse_args()
    cfg = load_config()
    if args.mock:
        m = ROOT / "outputs" / "mock"
        cfg["paths"].update(features=m / "features.csv",
                            perception=m / "per_image_metrics.csv",
                            real_rain_proxy=m / "real_rain_proxy.csv",
                            labels=m / "health_labels.csv",
                            model=m / "model_xgb.json",
                            results=m / "results")
    return cfg


def health_class(health, cfg):
    t = cfg["health_classes"]
    return np.select([health >= t["healthy"], health >= t["degraded"]],
                     CLASSES[:2], CLASSES[2])


def feature_cols(df, use_grid):
    """11 global features (+ 36 grid features nếu use_grid và features.csv có các cột đó)."""
    return FEATURES + ([c for c in GRID_FEATURES if c in df] if use_grid else [])


def prepare_X(df, cols=FEATURES):
    X = df[cols].astype(float).copy()
    logc = [c for c in cols if re.sub(r"_r\dc\d$", "", c) in LOG_FEATURES]
    X[logc] = np.log1p(X[logc].clip(lower=0))
    return X


def load_dataset(cfg):
    """features.csv JOIN health_labels.csv theo image_id."""
    feats = pd.read_csv(cfg["paths"]["features"])
    labels = pd.read_csv(cfg["paths"]["labels"])
    return feats.merge(labels, on="image_id", how="inner")
