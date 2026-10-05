"""[Member 3] Sinh dữ liệu GIẢ đúng schema để test pipeline khi chưa có output thật của M1/M2/M4.

Output (outputs/mock/): features.csv, per_image_metrics.csv, real_rain_proxy.csv
Chạy: python -m health_model.mock_data  rồi các script khác với --mock
"""
import numpy as np
import pandas as pd

from common.config import ROOT
from health_model.common import FEATURES

# (type, level) -> severity (mức giảm recall kỳ vọng)
VARIANTS = {("none", 0): 0.0,
            ("rain", 1): 0.08, ("rain", 2): 0.22, ("rain", 3): 0.42,
            ("blur", 1): 0.05, ("blur", 2): 0.15, ("blur", 3): 0.35, ("blur", 4): 0.6,
            ("noise", 1): 0.05, ("noise", 2): 0.2, ("noise", 3): 0.45,
            ("dark", 1): 0.15, ("dark", 2): 0.5,
            ("overexposure", 1): 0.12, ("overexposure", 2): 0.4}


def base_features(rng):
    bstd = rng.uniform(0.18, 0.26)
    return {"laplacian_variance": rng.lognormal(np.log(300), 0.5),
            "mean_brightness": rng.uniform(0.35, 0.55), "brightness_std": bstd,
            "contrast": bstd * 3.2 * rng.uniform(0.9, 1.1), "entropy": rng.uniform(7.0, 7.6),
            "saturation_ratio": rng.uniform(0, 0.02), "dark_pixel_ratio": rng.uniform(0.01, 0.05),
            "noise_estimate": rng.lognormal(np.log(3), 0.3), "edge_density": rng.uniform(0.05, 0.12),
            "gradient_magnitude": rng.lognormal(np.log(25), 0.3),
            "color_statistics": rng.uniform(20, 45)}


def degrade(f, kind, k, rng):
    f = dict(f)
    if kind == "blur":
        a = [0.5, 0.25, 0.08, 0.02][k - 1]
        f["laplacian_variance"] *= a; f["gradient_magnitude"] *= a ** 0.5
        f["edge_density"] *= a ** 0.6; f["noise_estimate"] *= 0.8; f["entropy"] -= 0.08 * k
    elif kind in ("noise",):
        s = [10, 25, 45][k - 1]
        f["noise_estimate"] += s; f["laplacian_variance"] += 4 * s ** 2
        f["gradient_magnitude"] += 0.8 * s; f["edge_density"] += 0.05 * k; f["entropy"] += 0.05
    elif kind == "dark":
        a = [0.35, 0.15][k - 1]
        f["mean_brightness"] *= a; f["brightness_std"] *= a; f["contrast"] *= a
        f["dark_pixel_ratio"] += [0.3, 0.65][k - 1]; f["entropy"] -= [0.6, 1.5][k - 1]
        f["laplacian_variance"] *= a ** 2; f["gradient_magnitude"] *= a
        f["edge_density"] *= a; f["color_statistics"] *= a
    elif kind == "overexposure":
        a = [0.45, 0.2][k - 1]
        f["mean_brightness"] = 1 - (1 - f["mean_brightness"]) * a
        f["saturation_ratio"] += [0.2, 0.5][k - 1]; f["contrast"] *= [0.7, 0.4][k - 1]
        f["brightness_std"] *= [0.8, 0.5][k - 1]; f["entropy"] -= [0.4, 1.0][k - 1]
        f["edge_density"] *= [0.8, 0.5][k - 1]; f["color_statistics"] *= [0.7, 0.4][k - 1]
    elif kind in ("rain", "real_rain"):
        a = [0.9, 0.75, 0.6][k - 1] if kind == "rain" else k   # real_rain: k liên tục
        f["contrast"] *= a; f["brightness_std"] *= a; f["color_statistics"] *= a
        f["laplacian_variance"] *= a ** 3; f["edge_density"] *= a; f["gradient_magnitude"] *= a
        f["mean_brightness"] += 0.15 * (1 - a); f["entropy"] -= 0.5 * (1 - a)
    return {k_: v * rng.uniform(0.95, 1.05) for k_, v in f.items()}


def main(n_parents=600, seed=0):
    rng = np.random.default_rng(seed)
    feats, perf, proxy = [], [], []
    for i in range(n_parents):
        pid = f"P{i:04d}_ref"
        split = rng.choice(["train", "val", "test"], p=[0.7, 0.15, 0.15])
        base, n_gt = base_features(rng), 2 + rng.poisson(8)
        r_clean = rng.uniform(0.8, 0.98)
        for (kind, k), sev in VARIANTS.items():
            iid = pid if kind == "none" else f"P{i:04d}_{kind}{k}"
            f = base if kind == "none" else degrade(base, kind, k, rng)
            feats.append({"image_id": iid, "parent_id": pid, "split": split, "condition": "rain_ref",
                          "degradation_type": kind, "degradation_level": k,
                          "is_synthetic": int(kind != "none"), **f})
            rec = np.clip(r_clean * (1 - sev * rng.uniform(0.6, 1.4)), 0, 1)
            tp = rng.binomial(n_gt, rec)
            fp = rng.poisson(1)
            perf.append({"image_id": iid, "n_gt": n_gt, "tp": tp, "fp": fp, "fn": n_gt - tp,
                         "precision": tp / max(tp + fp, 1), "recall": tp / n_gt})
        a = rng.uniform(0.55, 1.0)       # ảnh mưa thật cùng địa điểm
        iid = f"P{i:04d}_rain"
        feats.append({"image_id": iid, "parent_id": pid, "split": split, "condition": "rain",
                      "degradation_type": "real_rain", "degradation_level": 0, "is_synthetic": 0,
                      **degrade(base_features(rng), "real_rain", a, rng)})
        n_ref = 2 + rng.poisson(8)
        tp = rng.binomial(n_ref, np.clip(0.9 - 1.2 * (1 - a) + rng.normal(0, 0.08), 0, 1))
        proxy.append({"image_id": iid, "n_ref": n_ref, "tp": tp, "proxy_recall": tp / n_ref})

    out = ROOT / "outputs" / "mock"
    out.mkdir(parents=True, exist_ok=True)
    feats = pd.DataFrame(feats)
    feats[FEATURES] = feats[FEATURES].round(5)
    feats.to_csv(out / "features.csv", index=False)
    pd.DataFrame(perf).round(4).to_csv(out / "per_image_metrics.csv", index=False)
    pd.DataFrame(proxy).round(4).to_csv(out / "real_rain_proxy.csv", index=False)
    print(f"mock: {len(feats)} rows features.csv, {len(perf)} rows per_image_metrics.csv -> {out}")


if __name__ == "__main__":
    main()
