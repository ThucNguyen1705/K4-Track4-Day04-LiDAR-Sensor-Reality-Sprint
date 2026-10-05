"""[Member 3] Train XGBoost: 11 IQA features -> health in [0, 1].

Train trên split train, early stopping trên val (split do Member 1 chia, nằm trong features.csv).
Output: health_model/model/model_xgb.json, outputs/results/feature_importance.csv + plots/
"""
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from xgboost import XGBRegressor

from health_model.common import feature_cols, get_cfg, load_dataset, prepare_X


def build_model(cfg):
    params = dict(cfg["xgboost"])
    return XGBRegressor(objective="reg:squarederror", eval_metric="mae",
                        random_state=cfg["seed"], **params)


def fit(cfg, df, cols, verbose=False):
    tr, va = df[df["split"] == "train"], df[df["split"] == "val"]
    model = build_model(cfg)
    model.fit(prepare_X(tr, cols), tr["health"], eval_set=[(prepare_X(va, cols), va["health"])],
              verbose=verbose)
    return model


def features_path(model_path):
    return model_path.with_suffix(".features.json")


def main():
    cfg = get_cfg(__doc__)
    p = cfg["paths"]
    df = load_dataset(cfg)
    cols = feature_cols(df, cfg["model"]["use_grid"])
    print(f"train={(df['split'] == 'train').sum()}  val={(df['split'] == 'val').sum()}  "
          f"features={len(cols)}")

    model = fit(cfg, df, cols, verbose=200)
    print(f"best_iteration={model.best_iteration}  val_mae={model.best_score:.4f}")

    p["model"].parent.mkdir(parents=True, exist_ok=True)
    model.save_model(p["model"])
    json.dump(cols, open(features_path(p["model"]), "w"))
    print(f"saved -> {p['model']}")

    imp = (pd.Series(model.get_booster().get_score(importance_type="gain"))
           .reindex(cols).fillna(0).sort_values().tail(20))
    out = p["results"]
    (out / "plots").mkdir(parents=True, exist_ok=True)
    imp.rename("gain").to_csv(out / "feature_importance.csv", index_label="feature")
    imp.plot.barh(figsize=(7, 0.3 * len(imp) + 1.5), title="XGBoost feature importance (gain)")
    plt.tight_layout()
    plt.savefig(out / "plots" / "feature_importance.png", dpi=120)


if __name__ == "__main__":
    main()
