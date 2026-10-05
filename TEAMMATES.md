# TEAMMATES — K4 Track 4 Day 04 · Sensor Reality Sprint

**Đề tài:** Camera Health Assessment cho ADAS (IQA features → XGBoost → Health ∈ [0, 1])
**Tên nhóm:** `LiDAR` *(thư mục gốc repo: `K4-Track4-Day04-LiDAR-Sensor-Reality-Sprint`. Nếu tên nhóm khác thì đổi tên repo theo mẫu `K4-Track4-Day04-TenNhom-Sensor-Reality-Sprint`)*

## Thành viên (đúng 5 người)

| # | Họ và tên | MSSV | Vai trò | Thư mục phụ trách | Báo cáo riêng |
|---|---|---|---|---|---|
| 1 | Ngô Xuân Hoàng | 2A202602597 | Dataset + Data Pipeline | `data/`, `dataset/` | [reports/individual/2A202602597_NgoXuanHoang.md](reports/individual/2A202602597_NgoXuanHoang.md) |
| 2 | Phan Trọng Hoàn | 2A202602954 | Image Quality / Feature Engineering | `features/` | [reports/individual/2A202602954_PhanTrongHoan.md](reports/individual/2A202602954_PhanTrongHoan.md) |
| 3 | Phan Danh Đạt | 2A202602627 | ADAS Validation + System/Demo | `perception/`, `analysis/`, `demo/` | [reports/individual/2A202602627_PhanDanhDat.md](reports/individual/2A202602627_PhanDanhDat.md) |
| 4 | Nguyễn Đăng Thực | 2A202603014 | Ground Truth + XGBoost | `health_model/`, `kaggle/` | [reports/individual/2A202603014_NguyenDangThuc.md](reports/individual/2A202603014_NguyenDangThuc.md) |
| 5 | Lê Châu Trần Phát | 2A202602545 | Integration, Reproducibility & Documentation | `scripts/`, `docs/`, `reports/` | [reports/individual/2A202602545_LeChauTranPhat.md](reports/individual/2A202602545_LeChauTranPhat.md) |

Báo cáo nhóm (kết quả chung): [reports/REPORT.md](reports/REPORT.md)

## Phân công chi tiết

### 1. Ngô Xuân Hoàng — Dataset + Data Pipeline
- Tải và chuẩn hoá ACDC rain ([Kaggle](https://www.kaggle.com/datasets/njayadithya/rgb-anon-trainvaltest)): ảnh mưa thật + ảnh tham chiếu trời quang `*_ref`, resize 1280×720.
- Giữ split train/val/test theo ACDC (chia theo sequence, không leakage).
- Sinh degradation: rain L1–3, blur L1–4, noise L1–3, dark L1–2, overexposure L1–2 (`data/corruption/`).
- Xuất `dataset/metadata.csv` theo data contract trong README.
- **Bằng chứng:** `data/`, `kaggle/pipeline.py` (hàm `degrade`, `synth_rain`), ảnh `reports/figures/examples_*.png`, `reports/figures/diff_*.png`.

### 2. Phan Trọng Hoàn — Image Quality / Feature Engineering
- 11 IQA features toàn ảnh: Laplacian variance, mean brightness, brightness std, contrast, entropy, saturation ratio, dark pixel ratio, noise estimate (Immerkær), edge density, gradient magnitude, color statistics (colorfulness).
- 36 feature lưới 3×3 (`features/spatial.py`) cho degradation cục bộ.
- `features/extract_features.py` → `outputs/features/features.csv`.
- Phân tích độ nhạy feature theo từng loại/mức degradation.
- **Bằng chứng:** `features/`, `reports/figures/feature_response.png`, `reports/figures/feature_health_corr.png`.

### 3. Phan Danh Đạt — ADAS Validation + System/Demo
- Detector offline: YOLO11x tạo pseudo-GT trên ảnh sạch, YOLO11s là detector được đánh giá; matching IoU ≥ 0.5 cùng class.
- `outputs/perception/per_image_metrics.csv` (recall/precision theo ảnh), `real_rain_proxy.csv`.
- Phân tích tương quan Health ↔ Detection recall, ánh xạ Health → camera weight (`demo/adaptive_weight.py`), demo.
- **Bằng chứng:** `perception/`, `kaggle/pipeline.py` (hàm `match`, `boxes`), `reports/figures/real_rain*.png`, `health_model/predict.py`.

### 4. Nguyễn Đăng Thực — Ground Truth + XGBoost
- Thiết kế nhãn `health = recall(degraded) / recall(clean)` (`health_model/make_labels.py`), lọc ảnh có < 3 object.
- Train XGBoost (`train_xgboost.py`), so sánh với Rule-based / Linear Regression / Random Forest / LightGBM và XGBoost + grid (`evaluate.py`).
- Chạy pipeline trên Kaggle GPU (`kaggle/`), sản phẩm cuối `health_model/predict.py`.
- **Bằng chứng:** `outputs/results/results.csv`, `results_by_degradation.csv`, `outputs/logs/`, `reports/figures/`.

### 5. Lê Châu Trần Phát — Integration, Reproducibility & Documentation
- Chạy lại toàn bộ pipeline từ máy sạch theo README (`pip install -r requirements.txt` → các lệnh trong mục "Cách chạy"), ghi lại lỗi/thiếu sót, lưu log vào `outputs/logs/`.
- Viết `scripts/run_all.sh` chạy tuần tự: make_labels → train_xgboost → evaluate → make_figures.
- Viết `scripts/check_schema.py`: kiểm tra tên cột các CSV trong `outputs/` khớp data contract trong README.
- Tổng hợp nguồn tham khảo (paper/repo) vào `docs/references.md`: ACDC, YOLO11 (Ultralytics), XGBoost, LightGBM, Immerkær 1996 (noise), Hasler & Süsstrunk 2003 (colorfulness), ImageNet-C / nuScenes-C (corruption benchmark).
- Soát chính tả và link trong `README.md`, `reports/REPORT.md`, gom slide nhóm.
- Phụ trách checklist nộp bài bên dưới: kiểm tra đủ 5 bản báo cáo riêng và link repo mở được.
- **Bằng chứng:** `scripts/`, `docs/references.md`, log chạy lại trong `outputs/logs/`.

## Dùng chung repository, nộp riêng

- **Kết quả chung của nhóm:** code, cấu hình (`configs/`), dữ liệu được phép chia sẻ, log (`outputs/logs/`), plot (`reports/figures/`, `outputs/results/plots/`).
- **Bản riêng của từng người:** báo cáo/slide ngắn trong `reports/individual/<MSSV>_<HoTen>.md` (hoặc `.pdf`/`.pptx` cùng tên). Mỗi bản phải ghi rõ:
  - nguồn paper/repo đã dùng;
  - cách nhóm đo metric (nhãn health, MAE/R²/Spearman, F1);
  - failure case;
  - engineering decision mình đưa ra;
  - bảng/plot và **lệnh chạy chung** mà nội dung dựa vào (đường dẫn trong repo).
- Mỗi người nộp trên **VLearn**: bản riêng của mình + **cùng một URL repo nhóm**.

## Checklist trước khi nộp

- [ ] Thư mục gốc đặt tên `K4-Track4-Day04-TenNhom-Sensor-Reality-Sprint`
- [ ] `TEAMMATES.md` ở thư mục gốc, đúng **5** thành viên, đủ họ tên + MSSV
- [ ] Repo có mã nguồn + cách chạy, log/ảnh/plot đủ để kiểm tra metric
- [ ] **5** bản báo cáo/slide riêng trong `reports/individual/`, mỗi bản dẫn bảng/plot + lệnh chạy
- [ ] **5** lượt nộp trên VLearn, mỗi lượt có bản riêng đúng người + cùng URL repo
- [ ] Sau khi nộp: mở lại link, xác nhận người xem truy cập được bản riêng, `TEAMMATES.md`, metric, failure case, log/plot
- [ ] Đối chiếu rubric: **40% demo · 25% failure · 20% thuật toán · 15% trade-off**
