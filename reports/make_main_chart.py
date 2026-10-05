"""Biểu đồ chính cho slide/báo cáo: health theo mức lỗi — nhãn GT vs XGBoost vs baseline rule-based.

Nguồn: outputs/results/predictions.csv (test, 6 330 ảnh). Chạy: python -m reports.make_main_chart
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from common.config import load_config

INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
SERIES = [("health", "Nhãn GT (recall ratio)", INK, "-", "o"),
          ("pred_XGBoost", "XGBoost (nhóm)", "#2a78d6", "-", "s"),
          ("pred_RuleBased", "Rule-based (baseline)", "#eb6834", "--", "D")]
TYPES = [("rain", "Mưa tổng hợp"), ("blur", "Blur"), ("noise", "Noise"),
         ("dark", "Thiếu sáng"), ("overexposure", "Cháy sáng")]


def main():
    cfg = load_config()
    p = pd.read_csv(cfg["paths"]["results"] / "predictions.csv")
    clean = p[p["degradation_type"] == "none"]
    plt.rcParams.update({"font.size": 15, "axes.edgecolor": MUTED, "axes.labelcolor": MUTED,
                         "xtick.color": MUTED, "ytick.color": MUTED})
    fig, axes = plt.subplots(1, 5, figsize=(16, 5.6), sharey=True, facecolor=SURFACE)
    for ax, (t, name) in zip(axes, TYPES):
        g = pd.concat([clean, p[p["degradation_type"] == t]])
        m = g.groupby("degradation_level")[[s[0] for s in SERIES]].mean()
        ax.set_facecolor(SURFACE)
        for col, label, color, ls, mk in SERIES:
            ax.plot(m.index, m[col], ls, color=color, lw=2.5, marker=mk, ms=8,
                    markeredgecolor=SURFACE, markeredgewidth=2, label=label)
        ax.set_xticks(m.index, ["sạch"] + [f"L{k}" for k in m.index[1:]])
        ax.set_ylim(0, 1.05)
        ax.grid(axis="y", color=GRID, lw=1)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        last = m.index[-1]
        worst = g[g["degradation_level"] == last]
        mae = (worst["pred_XGBoost"] - worst["health"]).abs().mean()
        ax.set_title(f"{name}\nMAE XGBoost ở L{last}: {mae:.2f}", fontsize=16, color=INK, pad=10)
    axes[0].set_ylabel("Health (0–1, = recall / recall ảnh sạch)")
    fig.legend(*axes[0].get_legend_handles_labels(), loc="upper center", ncol=3, frameon=False,
               fontsize=15, bbox_to_anchor=(0.5, 1.0))
    fig.text(0.01, 0.01, "Nguồn: nhóm tự đo — ACDC rain *_ref (ảnh thật) + degradation tổng hợp, "
             "test 6 330 ảnh (422 ảnh/mức), YOLO11s vs pseudo-GT YOLO11x.\n"
             "Điểm = trung bình theo mức; MAE = sai số tuyệt đối trung bình từng ảnh.", fontsize=12, color=MUTED)
    fig.tight_layout(rect=(0, 0.08, 1, 0.92))
    out = cfg["paths"]["results"].parents[1] / "reports" / "figures" / "main_health_by_level.png"
    fig.savefig(out, dpi=120, facecolor=SURFACE)
    print(out)


if __name__ == "__main__":
    main()
