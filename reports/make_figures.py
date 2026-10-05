"""Sinh hình cho reports/REPORT.md từ output Kaggle + kết quả health_model.

Cần: outputs/kaggle/examples/ (giải nén examples.zip), outputs/features/features.csv,
     outputs/labels/health_labels.csv, outputs/results/predictions.csv, real_rain_predictions.csv
Chạy: python -m reports.make_figures
"""
import json
import shutil
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from common.config import load_config
from health_model.common import FEATURES

cfg = load_config()
P = cfg["paths"]
ROOT = P["results"].parents[1]
EX = ROOT / "outputs" / "kaggle" / "examples"
FIG = ROOT / "reports" / "figures"
ORDER = ["none_0", "rain_1", "rain_2", "rain_3", "blur_1", "blur_2", "blur_3", "blur_4",
         "noise_1", "noise_2", "noise_3", "dark_1", "dark_2", "overexposure_1", "overexposure_2"]
COCO = {0: "person", 1: "bicycle", 2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}


def rgb(path):
    return cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)


def load_example(d):
    info = json.load(open(d / "boxes.json"))
    vs = {f"{v['type']}_{v['level']}": (iid, v) for iid, v in info["variants"].items()}
    return info, vs


def draw(img, gt, v):
    im = img.copy()
    for b, ok in zip(gt["boxes"], v["gt_matched"]):
        x1, y1, x2, y2 = map(int, b)
        cv2.rectangle(im, (x1, y1), (x2, y2), (0, 220, 0) if ok else (255, 30, 30), 3 if ok else 4)
    for b, tp in zip(v["pred_boxes"], v["pred_tp"]):
        if not tp:
            x1, y1, x2, y2 = map(int, b)
            cv2.rectangle(im, (x1, y1), (x2, y2), (255, 210, 0), 2)
    return im


def fig_examples(d, preds):
    """Lưới 15 biến thể: box xanh = GT được phát hiện, đỏ = GT bị bỏ sót, vàng = false positive."""
    info, vs = load_example(d)
    gt = info["gt"]
    fig, axes = plt.subplots(4, 4, figsize=(20, 12.5))
    for ax in axes.ravel():
        ax.axis("off")
    for ax, key in zip(axes.ravel(), ORDER):
        iid, v = vs[key]
        ax.imshow(draw(rgb(d / f"{iid}.jpg"), gt, v))
        n, tp = len(v["gt_matched"]), sum(v["gt_matched"])
        title = f"{key.replace('_0', ' (clean)')}  recall {tp}/{n}"
        if iid in preds.index:
            r = preds.loc[iid]
            title += f"\nhealth GT={r['health']:.2f}  pred={r['pred_XGBoost']:.2f}"
        ax.set_title(title, fontsize=11)
    for k, (txt, col) in enumerate([("■ GT được YOLO11s phát hiện", "green"),
                                    ("■ GT bị bỏ sót (miss)", "red"),
                                    ("■ false positive", "#d4a000")]):
        axes[-1, -1].text(0.05, 0.7 - 0.2 * k, txt, color=col, fontsize=14)
    fig.suptitle(f"{d.name}: pseudo-GT YOLO11x trên ảnh sạch ({len(gt['boxes'])} object) — "
                 "xanh = phát hiện, đỏ = bỏ sót, vàng = FP", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out = FIG / f"examples_{d.name}.png"
    fig.savefig(out, dpi=80)
    plt.close(fig)
    return out


def fig_diff(d):
    """|degraded - clean| theo từng mức: ảnh khác ảnh sạch ở đâu và nhiều đến đâu."""
    info, vs = load_example(d)
    clean = rgb(d / f"{vs['none_0'][0]}.jpg").astype(np.float32)
    rows = [("rain", 3), ("blur", 4), ("noise", 3), ("dark", 2), ("overexposure", 2)]
    fig, axes = plt.subplots(len(rows), 4, figsize=(18, 12))
    for r, (t, nlev) in enumerate(rows):
        for c in range(4):
            ax = axes[r, c]
            ax.axis("off")
            if c >= nlev:
                continue
            iid, _ = vs[f"{t}_{c + 1}"]
            diff = np.abs(rgb(d / f"{iid}.jpg").astype(np.float32) - clean).mean(axis=2)
            ax.imshow(diff, cmap="magma", vmin=0, vmax=120)
            ax.set_title(f"{t} L{c + 1}: mean |Δ| = {diff.mean():.1f}", fontsize=11)
    fig.suptitle(f"{d.name}: bản đồ sai khác tuyệt đối so với ảnh sạch (0 = giống hệt, sáng = khác nhiều)",
                 fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out = FIG / f"diff_{d.name}.png"
    fig.savefig(out, dpi=80)
    plt.close(fig)
    return out


def fig_zoom(d):
    """Crop quanh object GT lớn nhất bị bỏ sót ở mức nặng: chi tiết bị mất thế nào."""
    info, vs = load_example(d)
    boxes = np.array(info["gt"]["boxes"])
    missed = np.zeros(len(boxes))
    for k in ORDER[1:]:
        missed += ~np.array(vs[k][1]["gt_matched"], bool)
    area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    i = int(np.argmax(area * (missed + 0.5)))
    x1, y1, x2, y2 = boxes[i]
    cx, cy, s = (x1 + x2) / 2, (y1 + y2) / 2, max(x2 - x1, y2 - y1) * 1.6 + 40
    X1, Y1 = int(max(cx - s / 2, 0)), int(max(cy - s / 2, 0))
    X2, Y2 = int(min(cx + s / 2, 1280)), int(min(cy + s / 2, 720))
    fig, axes = plt.subplots(3, 5, figsize=(17, 10.5))
    for ax, key in zip(axes.ravel(), ORDER):
        iid, v = vs[key]
        crop = rgb(d / f"{iid}.jpg")[Y1:Y2, X1:X2]
        ax.imshow(crop)
        ok = v["gt_matched"][i]
        ax.set_title(f"{key.replace('_0', ' (clean)')}: {'detected' if ok else 'MISSED'}",
                     color="green" if ok else "red", fontsize=12)
        ax.axis("off")
    fig.suptitle(f"{d.name}: zoom object #{i} ({COCO.get(info['gt']['cls'][i], '?')})", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out = FIG / f"zoom_{d.name}.png"
    fig.savefig(out, dpi=80)
    plt.close(fig)
    return out


def fig_feature_response(feats):
    """Median log2(feature / feature ảnh sạch cùng parent) theo từng loại/mức degradation."""
    syn = feats[feats["degradation_type"] != "real_rain"].copy()
    clean = syn[syn["degradation_type"] == "none"].set_index("image_id")[FEATURES]
    syn = syn[syn["degradation_type"] != "none"]
    ratio = np.log2((syn[FEATURES].to_numpy() + 1e-3) / (clean.loc[syn["parent_id"]].to_numpy() + 1e-3))
    r = pd.DataFrame(ratio, columns=FEATURES)
    r["variant"] = (syn["degradation_type"] + "_" + syn["degradation_level"].astype(str)).to_numpy()
    m = r.groupby("variant")[FEATURES].median().reindex([o for o in ORDER[1:] if o in set(r["variant"])])
    fig, ax = plt.subplots(figsize=(13, 7))
    im = ax.imshow(m.clip(-4, 4), cmap="RdBu_r", vmin=-4, vmax=4, aspect="auto")
    ax.set_xticks(range(len(FEATURES)), FEATURES, rotation=35, ha="right")
    ax.set_yticks(range(len(m)), m.index)
    for i in range(m.shape[0]):
        for j in range(m.shape[1]):
            ax.text(j, i, f"{m.iat[i, j]:.1f}", ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax, label="median log2(feature / clean)")
    ax.set_title("Feature thay đổi thế nào so với ảnh sạch (đỏ = tăng, xanh = giảm)")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out = FIG / "feature_response.png"
    fig.savefig(out, dpi=100)
    plt.close(fig)
    return out, m


def fig_feature_corr(df):
    rho = pd.Series({f: spearmanr(df[f], df["health"])[0] for f in FEATURES}).sort_values()
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.barh(rho.index, rho.values, color=np.where(rho > 0, "#3a7dc9", "#d0533a"))
    ax.axvline(0, color="k", lw=0.8)
    ax.set(xlabel="Spearman ρ với health", title="Tương quan từng feature với health (train+val+test)")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out = FIG / "feature_health_corr.png"
    fig.savefig(out, dpi=110)
    plt.close(fig)
    return out, rho


def fig_health_dist(df):
    df = df.assign(variant=df["degradation_type"] + "_" + df["degradation_level"].astype(str))
    order = [o for o in ORDER if o in set(df["variant"])]
    fig, ax = plt.subplots(figsize=(12, 4.5))
    ax.boxplot([df.loc[df["variant"] == o, "health"] for o in order], showfliers=False)
    ax.set_xticks(range(1, len(order) + 1), order, rotation=40, ha="right")
    ax.set(ylabel="health label", title="Phân bố nhãn health (recall / recall_clean) theo degradation")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out = FIG / "health_label_distribution.png"
    fig.savefig(out, dpi=110)
    plt.close(fig)
    return out


def fig_real_rain():
    rr = P["results"] / "real_rain_predictions.csv"
    imgs = sorted((EX / "real_rain").glob("*.jpg"))
    if not rr.exists() or not imgs:
        return None
    pred = pd.read_csv(rr).set_index("image_id")
    items = sorted([(pred.loc[p.stem, "pred_health"], p) for p in imgs if p.stem in pred.index])
    items = items[:4] + items[len(items) // 2 - 2:len(items) // 2 + 2] + items[-4:]
    fig, axes = plt.subplots(3, 4, figsize=(18, 8.5))
    for ax, (h, p) in zip(axes.ravel(), items):
        ax.imshow(rgb(p))
        pr = pred.loc[p.stem, "proxy_recall"]
        ax.set_title(f"pred health {h:.2f}   proxy recall {pr:.2f}" if pr == pr else
                     f"pred health {h:.2f}", fontsize=11)
        ax.axis("off")
    fig.suptitle("Ảnh mưa thật ACDC (test) — sắp theo health dự đoán (thấp → cao)", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out = FIG / "real_rain_examples.png"
    fig.savefig(out, dpi=80)
    plt.close(fig)
    return out


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    feats = pd.read_csv(P["features"])
    labels = pd.read_csv(P["labels"])
    df = feats.merge(labels, on="image_id")
    preds = pd.read_csv(P["results"] / "predictions.csv").set_index("image_id")

    for d in sorted(p for p in EX.iterdir() if (p / "boxes.json").exists()):
        print(fig_examples(d, preds), fig_diff(d), fig_zoom(d))
    _, m = fig_feature_response(feats)
    m.round(2).to_csv(FIG / "feature_response.csv")
    _, rho = fig_feature_corr(df)
    rho.round(3).rename("spearman").to_csv(FIG / "feature_health_corr.csv")
    fig_health_dist(df)
    fig_real_rain()
    for f in (P["results"] / "plots").glob("*.png"):
        shutil.copy(f, FIG / f.name)
    print(f"figures -> {FIG}")


if __name__ == "__main__":
    main()
