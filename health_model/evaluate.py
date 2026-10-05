"""[Member 3] Benchmark trên split test: RuleBased, LinearRegression, RandomForest, LightGBM, XGBoost.

Regression: MAE, RMSE, R2, Pearson, Spearman.
Classification (Healthy/Degraded/Critical): Accuracy, macro Precision/Recall/F1, confusion matrix.
Nếu có ảnh mưa thật (degradation_type == real_rain) + real_rain_proxy.csv: dự đoán health trên
ảnh thật (không có trong train) và tương quan với proxy recall.

Output (outputs/results/): results.csv, results_by_degradation.csv, predictions.csv,
real_rain_predictions.csv, plots/*.png
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import (accuracy_score, confusion_matrix, mean_absolute_error,
                             mean_squared_error, precision_recall_fscore_support, r2_score)
from xgboost import XGBRegressor

import json

from health_model.baselines import get_baselines
from health_model.common import (CLASSES, FEATURES, feature_cols, get_cfg, health_class,
                                 load_dataset, prepare_X)
from health_model.train_xgboost import features_path, fit


def score(y, pred, cfg):
    yc, pc = health_class(y, cfg), health_class(pred, cfg)
    p, r, f1, _ = precision_recall_fscore_support(yc, pc, labels=CLASSES, average="macro",
                                                  zero_division=0)
    return {"MAE": mean_absolute_error(y, pred),
            "RMSE": np.sqrt(mean_squared_error(y, pred)),
            "R2": r2_score(y, pred),
            "Pearson": pearsonr(y, pred)[0],
            "Spearman": spearmanr(y, pred)[0],
            "Accuracy": accuracy_score(yc, pc),
            "Precision": p, "Recall": r, "F1": f1}


def plot_confusion(y, pred, cfg, path):
    cm = confusion_matrix(health_class(y, cfg), health_class(pred, cfg), labels=CLASSES)
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ax.imshow(cm, cmap="Blues")
    for i in range(3):
        for j in range(3):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set(xticks=range(3), yticks=range(3), xticklabels=CLASSES, yticklabels=CLASSES,
           xlabel="Predicted", ylabel="True", title="XGBoost — test")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_pred_vs_true(test, path):
    fig, ax = plt.subplots(figsize=(5.5, 5))
    for t, g in test.groupby("degradation_type"):
        ax.scatter(g["health"], g["pred_XGBoost"], s=8, alpha=0.5, label=t)
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set(xlabel="True health", ylabel="Predicted health (XGBoost)", xlim=(0, 1.02),
           ylim=(0, 1.02), title="Predicted vs true — test")
    ax.legend(fontsize=8, markerscale=2)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_by_level(by_deg, path):
    fig, ax = plt.subplots(figsize=(9, 4))
    lbl = by_deg["degradation_type"] + "_" + by_deg["degradation_level"].astype(str)
    x = np.arange(len(by_deg))
    ax.bar(x - 0.2, by_deg["true_mean"], 0.4, label="true health")
    ax.bar(x + 0.2, by_deg["pred_mean"], 0.4, label="pred health (XGBoost)")
    ax.set_xticks(x, lbl, rotation=45, ha="right")
    ax.set(ylabel="mean health", title="Health theo loại / mức degradation — test")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def real_rain_eval(cfg, xgb, cols, test, out):
    feats = pd.read_csv(cfg["paths"]["features"])
    real = feats[feats["degradation_type"] == "real_rain"].copy()
    proxy_path = cfg["paths"]["real_rain_proxy"]
    if real.empty or not proxy_path.exists():
        return
    real["pred_health"] = np.clip(xgb.predict(prepare_X(real, cols)), 0, 1)
    real = real.merge(pd.read_csv(proxy_path), on="image_id", how="left")
    real.to_csv(out / "real_rain_predictions.csv", index=False)

    ok = real.dropna(subset=["proxy_recall"])
    ok = ok[ok["n_ref"] >= cfg["labels"]["min_gt"]]
    rho = spearmanr(ok["pred_health"], ok["proxy_recall"])[0] if len(ok) > 2 else np.nan
    clean = test[test["degradation_type"] == "none"]["pred_XGBoost"]
    print(f"\nReal rain: n={len(real)}  pred health mean={real['pred_health'].mean():.3f} "
          f"(clean test mean={clean.mean():.3f})  Spearman(pred, proxy_recall)={rho:.3f} "
          f"on n={len(ok)}")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    bins = np.linspace(0, 1, 26)
    axes[0].hist(clean, bins, alpha=0.6, density=True, label="clean (ref, test)")
    axes[0].hist(real["pred_health"], bins, alpha=0.6, density=True, label="real rain")
    axes[0].set(xlabel="predicted health", title="Phân bố health dự đoán")
    axes[0].legend()
    axes[1].scatter(ok["pred_health"], ok["proxy_recall"], s=10, alpha=0.6)
    axes[1].set(xlabel="predicted health", ylabel="proxy recall (YOLO11s vs YOLO11x)",
                title=f"Real rain — Spearman={rho:.2f}")
    fig.tight_layout()
    fig.savefig(out / "plots" / "real_rain.png", dpi=120)
    plt.close(fig)


def main():
    cfg = get_cfg(__doc__)
    p = cfg["paths"]
    out = p["results"]
    (out / "plots").mkdir(parents=True, exist_ok=True)

    df = load_dataset(cfg)
    tr = df[df["split"] == "train"]
    test = df[df["split"] == "test"].copy()
    X_tr, X_te = prepare_X(tr), prepare_X(test)

    models = get_baselines(cfg["seed"])
    for name, m in models.items():
        m.fit(X_tr, tr["health"])
    preds = {name: m.predict(X_te) for name, m in models.items()}
    n_feat = dict.fromkeys(preds, len(FEATURES))

    # XGBoost chính (đã train bởi train_xgboost.py) + biến thể còn lại (global <-> global+grid)
    xgb = XGBRegressor()
    xgb.load_model(p["model"])
    cols = json.load(open(features_path(p["model"])))
    preds["XGBoost"], n_feat["XGBoost"] = xgb.predict(prepare_X(test, cols)), len(cols)
    alt = FEATURES if len(cols) > len(FEATURES) else feature_cols(df, use_grid=True)
    if alt != cols:
        name = "XGBoost_grid" if len(alt) > len(FEATURES) else "XGBoost_global"
        preds[name], n_feat[name] = fit(cfg, df, alt).predict(prepare_X(test, alt)), len(alt)

    rows = []
    for name, pred in preds.items():
        test[f"pred_{name}"] = np.clip(pred, 0, 1)
        rows.append({"model": name, "n_features": n_feat[name],
                     **score(test["health"], test[f"pred_{name}"], cfg)})
    results = pd.DataFrame(rows).round(4)
    results.to_csv(out / "results.csv", index=False)
    print(f"test n={len(test)}\n{results.to_string(index=False)}")

    by_deg = (test.groupby(["degradation_type", "degradation_level"])
              .apply(lambda g: pd.Series({
                  "n": len(g), "true_mean": g["health"].mean(),
                  "pred_mean": g["pred_XGBoost"].mean(),
                  "MAE": mean_absolute_error(g["health"], g["pred_XGBoost"])}),
                     include_groups=False)
              .reset_index().round(4))
    by_deg.to_csv(out / "results_by_degradation.csv", index=False)
    print(f"\nXGBoost theo degradation:\n{by_deg.to_string(index=False)}")

    keep = ["image_id", "parent_id", "condition", "degradation_type", "degradation_level", "health"]
    test[keep + [c for c in test if c.startswith("pred_")]].to_csv(
        out / "predictions.csv", index=False)

    plot_pred_vs_true(test, out / "plots" / "pred_vs_true.png")
    plot_confusion(test["health"], test["pred_XGBoost"], cfg, out / "plots" / "confusion_matrix.png")
    plot_by_level(by_deg, out / "plots" / "health_by_degradation.png")
    real_rain_eval(cfg, xgb, cols, test, out)


if __name__ == "__main__":
    main()
