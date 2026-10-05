# Track-4-Camera Health Assessment

📄 **Báo cáo kết quả:** [reports/REPORT.md](reports/REPORT.md) · 👥 **Thành viên:** [TEAMMATES.md](TEAMMATES.md)

Đánh giá "sức khoẻ" camera cho ADAS: **image-quality features → XGBoost → Health ∈ [0, 1]**.
Detector (YOLO11s / RT-DETR) chỉ chạy **offline** để tạo và kiểm chứng nhãn Health.

```
Dataset (M1) → IQA Features (M2) → XGBoost Health (M3) → ADAS Validation + Demo (M4)
```

## Cấu trúc & phân công

```
.
├── configs/config.yaml        # path + tham số dùng chung (mọi người đọc từ đây)
├── common/config.py           # load_config()
│
├── data/                      # 👤 Member 1 — Dataset & Data Pipeline
│   ├── download/download.py
│   ├── preprocessing/preprocess.py
│   ├── preprocessing/split.py
│   ├── corruption/corrupt.py
│   └── build_metadata.py
├── dataset/                   # output của M1 (không commit ảnh, chỉ commit metadata.csv)
│   ├── train/ val/ test/
│   ├── corrupted/<type>/<level>/
│   └── metadata.csv
│
├── features/                  # 👤 Member 2 — IQA Feature Engineering
│   ├── blur.py  brightness.py  entropy.py  noise.py  edges.py  color.py
│   ├── spatial.py             # lưới 3x3
│   ├── extractor.py           # extract(img) -> 11 global + 36 grid
│   ├── extract_features.py    # → outputs/features/features.csv
│   └── crosscheck.py          # so với rain_features_csv/ của teammate
│
├── health_model/              # 👤 Member 3 — Ground Truth + XGBoost
│   ├── make_labels.py         # → outputs/labels/health_labels.csv
│   ├── baselines.py           # Rule-based, LinReg, RF, LightGBM
│   ├── train_xgboost.py
│   ├── evaluate.py            # → outputs/results/
│   ├── predict.py             # sản phẩm cuối: ảnh -> health / class / camera weight
│   ├── mock_data.py           # dữ liệu giả để test (--mock)
│   └── model/model_xgb.json
│
├── perception/                # 👤 Member 4 — ADAS Validation
│   ├── detector.py
│   ├── metrics.py
│   └── evaluate.py            # → outputs/perception/per_image_metrics.csv
├── analysis/                  # 👤 Member 4
│   ├── correlation.py         # Health vs Recall
│   └── visualization.py
├── demo/                      # 👤 Member 4
│   ├── adaptive_weight.py     # Health → camera weight
│   └── app.py
│
├── kaggle/                    # pipeline.py + build.py -> kernel Kaggle GPU (ACDC rain)
├── reports/                   # REPORT.md, figures/, make_figures.py, individual/ (5 báo cáo riêng)
├── rain_features_csv/         # features ảnh mưa thật do teammate tính (kiểm chéo)
├── outputs/                   # file trung gian giữa các module + logs/
│   ├── features/  perception/  labels/
│   └── results/ (results.csv, predictions.csv, plots/)
└── notebooks/                 # thử nghiệm, đặt tên <member>_<topic>.ipynb
```

## Data contract (file giao nhau giữa các member)

Khoá chung của mọi file là **`image_id`**. Đổi schema phải báo cả nhóm.

| File | Người tạo | Người dùng | Cột |
|---|---|---|---|
| `dataset/metadata.csv` | M1 | tất cả | `image_id, file_path, parent_id, scene_id, camera, condition, degradation_type, degradation_level, day_night, weather, split` |
| `outputs/features/features.csv` | M2 | M3, M4 | `image_id, parent_id, split, condition, degradation_type, degradation_level, is_synthetic, laplacian_variance, mean_brightness, brightness_std, contrast, entropy, saturation_ratio, dark_pixel_ratio, noise_estimate, edge_density, gradient_magnitude, color_statistics` (+ tuỳ chọn cột lưới `<feature>_r{row}c{col}`) |
| `outputs/perception/per_image_metrics.csv` | M4 | M3 | `image_id, n_gt, tp, fp, fn, precision, recall` |
| `outputs/perception/real_rain_proxy.csv` | M4 | M3 | `image_id, n_ref, tp, fp, proxy_recall` (ảnh mưa thật, không có ảnh clean khớp pixel) |
| `outputs/labels/health_labels.csv` | M3 | M3, M4 | `image_id, n_gt, recall, recall_clean, health, health_class` |
| `outputs/results/predictions.csv` | M3 | M4 | `image_id, parent_id, condition, degradation_type, degradation_level, health, pred_<Model>` |

Ghi chú:
- `parent_id`: id của ảnh clean gốc (ảnh clean thì bằng chính `image_id`) — cần để tính `health = recall / recall_clean`.
- `degradation_type ∈ {none, rain, blur, noise, dark, overexposure, real_rain}`, `degradation_level` là số nguyên (0 = clean); `camera` (vd. `CAM_FRONT`) dùng cho adaptive weighting.
- Split chia theo `scene_id` và **chỉ M1 tạo**; ảnh corrupted thuộc cùng split với ảnh gốc.
- Detector chạy 1 lần ở `perception/` (M4); M3 dùng kết quả đó để tạo nhãn, không chạy lại detector.

## Cách chạy

```bash
pip install -r requirements.txt

# chạy từ thư mục gốc repo
python -m data.build_metadata
python -m features.extract_features
python -m perception.evaluate
python -m health_model.make_labels      # thêm --mock để chạy trên dữ liệu giả (health_model/mock_data.py)
python -m health_model.train_xgboost
python -m health_model.evaluate
python -m analysis.correlation
streamlit run demo/app.py
```

Trong script, lấy path qua config:

```python
from common.config import load_config
cfg = load_config()
meta_path = cfg["paths"]["metadata"]
```

## Member 3 — chạy trên ACDC rain (Kaggle GPU)

`kaggle/pipeline.py` (dùng package `features/` của Member 2) chạy trên Kaggle (T4) với dataset
[njayadithya/rgb-anon-trainvaltest](https://www.kaggle.com/datasets/njayadithya/rgb-anon-trainvaltest)
và sinh `features.csv`, `per_image_metrics.csv`, `real_rain_proxy.csv`:

- Ảnh clean = `rain/*_ref` (trời quang), pseudo-GT = YOLO11x; degraded = rain/blur/noise/dark/overexposure
  sinh pixel-aligned; detector đánh giá = YOLO11s → `health = recall / recall_clean`.
- Ảnh mưa thật không có ảnh clean khớp pixel → chỉ dùng để kiểm chứng (proxy recall YOLO11s vs YOLO11x).
- Phần degradation/detector trong script là bản tạm; M1/M4 thay bằng module chính thức, giữ schema.
- Lệnh tái lập đầy đủ + bằng chứng: [reports/REPORT.md §10](reports/REPORT.md#10-tái-lập-kết-quả).

```bash
# KAGGLE_API_TOKEN đặt trong .env (không commit)
set -a && . ./.env && set +a
python kaggle/build.py                       # nhúng package features/ vào kernel
kaggle kernels push -p kaggle/kernel --accelerator NvidiaTeslaT4
kaggle kernels output nguyendangthuc11/camera-health-acdc-rain -p outputs/kaggle
```

## Quy ước làm việc

- Mỗi người làm trên branch riêng: `m1/data`, `m2/features`, `m3/health-model`, `m4/perception`; merge vào `main` qua PR.
- Chỉ sửa trong thư mục của mình; sửa file dùng chung (`configs/`, `common/`, README) thì báo nhóm.
- Không commit ảnh, weights (`*.pt`) hay CSV trung gian lớn — đã có trong `.gitignore`.
