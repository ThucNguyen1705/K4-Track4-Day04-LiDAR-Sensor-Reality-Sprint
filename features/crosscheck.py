"""[Member 2] Kiểm chéo features.csv của nhóm với bản tính độc lập của teammate (rain_features_csv/).

rain_features_csv/<split>/<sequence>/features.csv: 11 feature trên ảnh mưa thật ACDC, ảnh gốc 1920x1080,
mean_brightness / brightness_std ở thang 0-255. So với outputs/features/features.csv (1280x720, thang 0-1):
  1. Spearman / Pearson / tỉ lệ median từng feature trên cùng ảnh.
  2. Dự đoán health của XGBoost khi đưa feature teammate vào (chỉ đổi đơn vị brightness)
     -> đo độ nhạy của model với quy ước tính feature

Output: outputs/results/feature_crosscheck.csv, real_rain_teammate_predictions.csv, plots/feature_crosscheck.png
"""
import glob
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from xgboost import XGBRegressor

from common.config import ROOT, load_config
from features.extractor import FEATURES
from health_model.common import prepare_X
from health_model.train_xgboost import features_path


def load_teammate():
    t = pd.concat([pd.read_csv(f) for f in glob.glob(str(ROOT / "rain_features_csv/*/*/features.csv"))],
                  ignore_index=True)
    t["image_id"] = t["image_id"].str.extract(r"([^/]*?_frame_\d+)")[0] + "_rain"
    t[["mean_brightness", "brightness_std"]] /= 255
    return t


def main():
    cfg = load_config()
    out = cfg["paths"]["results"]
    ours = pd.read_csv(cfg["paths"]["features"])
    ours = ours[ours["degradation_type"] == "real_rain"]
    j = load_teammate().merge(ours, on="image_id", suffixes=("_tm", ""))
    print(f"matched {len(j)} ảnh mưa thật")

    cmp = pd.DataFrame([{"feature": f, "median_teammate": j[f + "_tm"].median(), "median_ours": j[f].median(),
                         "ratio_median": (j[f + "_tm"] / j[f].replace(0, np.nan)).median(),
                         "spearman": spearmanr(j[f + "_tm"], j[f])[0],
                         "pearson": np.corrcoef(j[f + "_tm"], j[f])[0, 1]} for f in FEATURES]).round(4)
    cmp.to_csv(out / "feature_crosscheck.csv", index=False)
    print(cmp.to_string(index=False))

    model = XGBRegressor()
    model.load_model(cfg["paths"]["model"])
    cols = json.load(open(features_path(cfg["paths"]["model"])))
    tm = j[[f + "_tm" for f in FEATURES]].set_axis(FEATURES, axis=1)
    j["pred_ours"] = np.clip(model.predict(prepare_X(j, cols)), 0, 1)
    j["pred_teammate"] = np.clip(model.predict(prepare_X(tm, cols)), 0, 1)
    j[["image_id", "split", "pred_ours", "pred_teammate"]].round(4).to_csv(
        out / "real_rain_teammate_predictions.csv", index=False)
    rho = spearmanr(j["pred_ours"], j["pred_teammate"])[0]
    gap = (j["pred_teammate"] - j["pred_ours"]).abs()
    print(f"health dự đoán: ours mean={j['pred_ours'].mean():.3f}  teammate mean="
          f"{j['pred_teammate'].mean():.3f}  |Δ| mean={gap.mean():.3f}  Spearman={rho:.3f}")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3))
    axes[0].barh(cmp["feature"], cmp["spearman"], color="#3a7dc9")
    axes[0].set(xlim=(0.7, 1.0), xlabel="Spearman (teammate vs ours, cùng ảnh)",
                title="Kiểm chéo 11 feature trên 1000 ảnh mưa thật")
    axes[1].scatter(j["pred_ours"], j["pred_teammate"], s=8, alpha=0.5)
    axes[1].plot([0, 1], [0, 1], "k--", lw=1)
    axes[1].set(xlabel="health dự đoán — feature của nhóm (1280x720)",
                ylabel="health dự đoán — feature teammate (1920x1080)",
                title=f"Độ nhạy model với quy ước feature (|Δ| TB={gap.mean():.2f})")
    fig.tight_layout()
    fig.savefig(out / "plots" / "feature_crosscheck.png", dpi=110)


if __name__ == "__main__":
    main()
