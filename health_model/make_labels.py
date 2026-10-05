"""[Member 3] Tạo nhãn Health từ kết quả detector (outputs/perception/per_image_metrics.csv).

health = recall(ảnh degraded) / recall(ảnh clean gốc = parent_id), clip về [0, 1].
Ảnh có quá ít object (n_gt < labels.min_gt) hoặc recall_clean = 0 bị loại vì nhãn không ổn định.

Output: outputs/labels/health_labels.csv
  image_id, n_gt, recall, recall_clean, health, health_class
"""
import pandas as pd

from health_model.common import get_cfg, health_class


def make_labels(features, perf, min_gt):
    parent = features.set_index("image_id")["parent_id"]
    df = perf.copy()
    df["parent_id"] = df["image_id"].map(parent).fillna(df["image_id"])
    df["recall_clean"] = df["parent_id"].map(perf.set_index("image_id")["recall"])
    df = df[(df["n_gt"] >= min_gt) & (df["recall_clean"] > 0)].copy()
    df["health"] = (df["recall"] / df["recall_clean"]).clip(0, 1)
    return df[["image_id", "n_gt", "recall", "recall_clean", "health"]]


def main():
    cfg = get_cfg(__doc__)
    p = cfg["paths"]
    labels = make_labels(pd.read_csv(p["features"]), pd.read_csv(p["perception"]),
                         cfg["labels"]["min_gt"])
    labels["health_class"] = health_class(labels["health"], cfg)
    p["labels"].parent.mkdir(parents=True, exist_ok=True)
    labels.to_csv(p["labels"], index=False)
    print(f"{len(labels)} labels -> {p['labels']}")
    print(labels["health"].describe().round(3).to_string())
    print(labels["health_class"].value_counts().to_string())


if __name__ == "__main__":
    main()
