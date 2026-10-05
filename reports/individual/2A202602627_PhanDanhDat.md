# Báo cáo cá nhân — Phan Danh Đạt (2A202602627)

**Vai trò:** ADAS Validation + System/Demo · **Repo nhóm:** [https://github.com/ThucNguyen1705/K4-Track4-Day04-LiDAR-Sensor-Reality-Sprint](https://github.com/ThucNguyen1705/K4-Track4-Day04-LiDAR-Sensor-Reality-Sprint) · **Báo cáo nhóm:** [../REPORT.md](../REPORT.md)

---

## 1. Phần việc của tôi

Tôi chịu trách nhiệm phân hệ **ADAS Validation, Phân tích tương quan Perception và Hệ thống Demo / Adaptive Weighting**, phụ trách chính các thư mục [`perception/`](../../perception/), [`analysis/`](../../analysis/), [`demo/`](../../demo/) cùng các module liên quan trên Kaggle GPU:

| Module / Script | File cụ thể | Vai trò và nhiệm vụ chính |
|---|---|---|
| **Perception Detector** | [`perception/detector.py`](../../perception/detector.py) | Xây dựng pipeline chạy Object Detector offline: dùng YOLO11x (conf 0.40, min-area $24 \times 24$ px) trích xuất Pseudo-Ground Truth trên ảnh sạch (`*_ref`); dùng YOLO11s (conf 0.25) làm detector được đánh giá trên 15 biến thể. Lọc 6 lớp ADAS quan trọng (COCO: person, bicycle, car, motorcycle, bus, truck). |
| **Perception Metrics** | [`perception/metrics.py`](../../perception/metrics.py) | Cài đặt thuật toán Greedy Matching theo thứ tự confidence giảm dần với ràng buộc cùng class và $\text{IoU} \ge 0.5$; tính $TP, FP, FN$, Precision, Recall theo từng frame; định nghĩa proxy recall cho ảnh mưa thật không có cặp ảnh trời quang đối chứng. |
| **Perception Evaluation** | [`perception/evaluate.py`](../../perception/evaluate.py) | Tổng hợp kết quả detector theo từng ảnh ([`per_image_metrics.csv`](../../outputs/perception/per_image_metrics.csv)) và tổng hợp bảng suy giảm hiệu năng theo từng điều kiện môi trường ([`per_condition_metrics.csv`](../../outputs/perception/per_condition_metrics.csv)). |
| **Kaggle GPU Pipeline** | [`kaggle/pipeline.py`](../../kaggle/pipeline.py) | Tích hợp detector, matching logic và trích xuất dữ liệu batch trên Kaggle T4 (~15 000 biến thể ảnh tổng hợp + 1 000 ảnh mưa thật ACDC). |
| **Phân tích tương quan** | [`analysis/correlation.py`](../../analysis/correlation.py) | Đánh giá tương quan giữa Camera Health dự đoán (XGBoost) và Detection Recall thực tế; phân rã Pearson và Spearman theo từng loại suy biến; phân tích định lượng Domain Gap trên ảnh mưa thật ([`correlation_analysis.csv`](../../outputs/results/correlation_analysis.csv)). |
| **Trực quan hoá ADAS** | [`analysis/visualization.py`](../../analysis/visualization.py) | Sinh biểu đồ phân tán Health vs Recall ([`scatter_health_recall.png`](../figures/scatter_health_recall.png)) và biểu đồ cột suy giảm năng lực nhận diện ADAS theo từng cấp độ ([`recall_by_condition.png`](../figures/recall_by_condition.png)). |
| **Adaptive Weighting** | [`demo/adaptive_weight.py`](../../demo/adaptive_weight.py) | Xây dựng cơ chế ánh xạ từ Camera Health score sang trọng số camera trong hệ thống Multi-Camera / Multi-Sensor Fusion ADAS ($w = h$ hoặc phi tuyến); cơ chế phát hiện suy biến và fallback an toàn sang LiDAR / Radar. |
| **Interactive Demo** | [`demo/app.py`](../../demo/app.py) | Xây dựng dashboard tương tác (Streamlit + CLI fallback) mô phỏng đánh giá sức khoẻ camera theo thời gian thực, điều phối trọng số 4 camera quanh xe (Front, Left, Right, Rear) và giải thích Failure Case. |

---

## 2. Nguồn tham khảo (paper / repo)

1. **ACDC Dataset:** Christos Sakaridis, Dengxin Dai, Luc Van Gool. *ACDC: The Adverse Conditions Dataset with Correspondences.* IEEE/CVF International Conference on Computer Vision (ICCV), 2021.  
   - Cung cấp tập dữ liệu ảnh lái xe thực tế với các cặp ảnh cùng hành trình trong điều kiện khắc nghiệt (Rain, Fog, Snow, Night) và ảnh đối chứng trời quang (`*_ref`).
2. **Ultralytics YOLO11:** Glenn Jocher et al. *Ultralytics YOLO11*, 2024. Repo: [github.com/ultralytics/ultralytics](https://github.com/ultralytics/ultralytics).  
   - Sử dụng YOLO11x làm teacher/pseudo-GT detector trên ảnh chuẩn và YOLO11s làm student/target detector chạy trên ảnh suy biến.
3. **Robustness & Synthetic Corruption Benchmarks:** Dan Hendrycks, Thomas Dietterich. *Benchmarking Neural Network Robustness to Common Corruptions and Perturbations* (ImageNet-C). ICLR 2019.  
   - Cơ sở lý thuyết về việc mô hình hoá các dạng suy biến cảm biến phổ biến: Gaussian noise, Defocus/Motion blur, Rain streaks, Brightness/Exposure.
4. **Sensor Failure & Multi-Modal Fusion trong ADAS:**  
   - Nghiên cứu về kiến trúc Camera-LiDAR Adaptive Late Fusion: khi chất lượng tín hiệu camera suy giảm, hệ thống phải giảm độ tin cậy ($w_{\text{cam}} \to 0$) và chuyển quyền ưu tiên cho cảm biến hình học tích cực (LiDAR / 4D Imaging Radar).
5. **XGBoost:** Tianqi Chen, Carlos Guestrin. *XGBoost: A Scalable Tree Boosting System.* ACM KDD, 2016.  
   - Mô hình Gradient Boosted Trees dùng để hồi quy điểm Health liên tục $\in [0, 1]$ từ 11 đặc trưng IQA.

---

## 3. Cách nhóm đo metric

### 3.1. Định nghĩa nhãn Ground Truth cho Camera Health
Thay vì đánh giá chất lượng ảnh bằng các độ đo thị giác con người (PSNR, SSIM - vốn không tương quan với năng lực AI), nhóm tôi định nghĩa Camera Health dựa trên **năng lực nhận diện thực tế của Perception Module (ADAS-centric definition)**:

$$\text{Health} = \text{clip}\left(\frac{\text{Recall}_{\text{YOLO11s}}(\text{ảnh suy biến})}{\text{Recall}_{\text{YOLO11s}}(\text{ảnh sạch cùng cảnh})}, 0.0, 1.0\right)$$

Trong đó:
- **Pseudo-Ground Truth:** YOLO11x chạy trên ảnh sạch (`*_ref`) với $\text{conf} = 0.40$, lọc bỏ box quá nhỏ ($S < 24 \times 24 = 576\text{ px}^2$). Số lượng vật thể là $n_{\text{gt}}$.
- **Detection Matching:** YOLO11s chạy trên từng biến thể với $\text{conf} = 0.25$. Dùng thuật toán Greedy Matching với $\text{IoU} \ge 0.5$ và trùng COCO class để xác định $TP, FP, FN$.
- **Lọc dữ liệu:** Loại bỏ ảnh có $n_{\text{gt}} < 3$ hoặc $\text{Recall}_{\text{clean}} = 0$ để tránh mẫu số bằng 0 và giảm nhiễu lượng tử hoá (còn lại 11 280 mẫu).

Phân lớp vận hành ADAS:
- **Healthy:** $\text{Health} \ge 0.8$ (Hoạt động bình thường, trọng số tin cậy tối đa).
- **Degraded:** $0.5 \le \text{Health} < 0.8$ (Cảnh báo suy giảm, giảm trọng số fusion, yêu cầu quét bù từ LiDAR).
- **Critical:** $\text{Health} < 0.5$ (Không tin cậy, ngắt camera khỏi perception pipeline, kích hoạt fallback sang LiDAR/Radar).

### 3.2. Kết quả đo lường và So sánh Benchmark trên Test Set (6 330 ảnh)
Kết quả đánh giá độc lập trên split test (chia theo sequence của ACDC, hoàn toàn không rò rỉ cảnh):

| Mô hình | Số Feature | MAE ↓ | RMSE ↓ | $R^2$ ↑ | Pearson ↑ | Spearman ↑ | Accuracy | Macro-F1 |
|---|---|---|---|---|---|---|---|---|
| **Rule-based (Khoảng cách Euclidean tới clean)** | 11 | 0.2543 | 0.3600 | -0.2397 | 0.0154 | 0.0664 | 0.5226 | 0.3078 |
| **Linear Regression** | 11 | 0.1785 | 0.2299 | 0.4943 | 0.7118 | 0.6745 | 0.6264 | 0.6068 |
| **Random Forest** | 11 | **0.1444** | 0.2017 | 0.6108 | 0.7936 | **0.7467** | 0.7226 | 0.6777 |
| **LightGBM** | 11 | 0.1477 | 0.2057 | 0.5952 | 0.7877 | 0.7349 | 0.7172 | 0.6643 |
| **XGBoost (Mô hình lựa chọn)** | 11 | **0.1455** | **0.1991** | **0.6207** | **0.7964** | **0.7460** | **0.7329** | **0.6857** |
| **XGBoost + Grid 3x3** | 47 | 0.1548 | 0.2079 | 0.5866 | 0.7870 | 0.7379 | 0.7118 | 0.6713 |

*Nguồn dẫn chứng:* File [`outputs/results/results.csv`](../../outputs/results/results.csv), biểu đồ ma trận nhầm lẫn [`reports/figures/confusion_matrix.png`](../figures/confusion_matrix.png), biểu đồ dự đoán vs thực tế [`reports/figures/pred_vs_true.png`](../figures/pred_vs_true.png).

### 3.3. Phân tích Tương quan Health vs Detection Recall thực tế
Thông qua script [`analysis/correlation.py`](../../analysis/correlation.py), tôi đã kiểm chứng mức độ gắn kết giữa Camera Health mà XGBoost dự đoán và Detection Recall thực tế của YOLO11s:

| Nhóm điều kiện | Số ảnh | Pearson ($\text{Pred}, \text{Recall}$) | Spearman ($\text{Pred}, \text{Recall}$) | Spearman ($\text{Pred}, \text{GT}$) |
|---|---|---|---|---|
| **Toàn bộ Test set (ALL)** | 6 330 | **0.7715** | **0.6870** | **0.7175** |
| Rain tổng hợp | 1 266 | 0.7816 | 0.7384 | 0.7540 |
| Blur | 1 688 | 0.7608 | 0.7223 | 0.7417 |
| Noise | 1 266 | 0.7286 | 0.6990 | 0.7139 |
| Dark | 844 | 0.4399 | 0.4133 | 0.4332 |
| Overexposure | 844 | 0.1972 | 0.1453 | 0.2553 |
| Clean (none) | 422 | -0.0247 | -0.0211 | 0.0500 |
| **Real Rain ACDC (OOD)** | 932 | **-0.0552** | **-0.0302** | *N/A (Không có GT)* |

*Dẫn chứng trực quan:* File [`reports/figures/scatter_health_recall.png`](../figures/scatter_health_recall.png) và [`outputs/results/correlation_analysis.csv`](../../outputs/results/correlation_analysis.csv).

---

## 4. Failure cases

Trong vai trò ADAS Validation & System Demo, tôi đã tiến hành phân tích sâu 4 Failure Cases chính của hệ thống:

```
+-----------------------------------------------------------------------------------------+
|                                    BẢN ĐỒ FAILURE CASES                                 |
+-----------------------------------------------------------------------------------------+
|  F1: Domain Gap trên Ảnh Mưa Thật ACDC                                                  |
|      - Hiện tượng: Model chấm trung bình 0.966 (99.9% Healthy), proxy recall < 0.7 ở 11%|
|      - Bản chất: Cần gạt nước che cục bộ + đường ướt phản chiếu; IQA toàn cảnh bị "mù"  |
+-----------------------------------------------------------------------------------------+
|  F2: Lượng tử hoá rời rạc khi ảnh có ít vật thể                                         |
|      - Hiện tượng: Ảnh có 3 object khiến recall nhảy bậc (0, 0.33, 0.67, 1.0)           |
|      - Bản chất: Nhãn bị nhiễu cục bộ, model dự đoán đúng xu hướng nhưng nhãn nhảy vọt  |
+-----------------------------------------------------------------------------------------+
|  F3: Giới hạn của Rule-based & Phi tuyến tính                                           |
|      - Hiện tượng: Rule-based đạt Spearman 0.066, R² = -0.24 (hoàn toàn vô dụng)        |
|      - Bản chất: Feature phản ứng 2 chiều trái ngược (Noise làm nét giả, Blur làm mờ)  |
+-----------------------------------------------------------------------------------------+
|  F4: Vùng suy biến trung gian & Lớp Degraded                                            |
|      - Hiện tượng: Lớp Degraded có F1 thấp nhất (0.46), 743 ảnh Healthy bị hạ nhầm      |
|      - Bản chất: Vùng ranh giới nhấp nháy của detector (Rain L2, Blur L3, Dark L2)     |
+-----------------------------------------------------------------------------------------+
```

### Failure Case 1: "Mù" trước ảnh mưa thật ACDC (Domain Gap Synthetic $\to$ Real)
- **Mô tả chi tiết:** Khi đưa 1 000 ảnh mưa thật của tập ACDC vào mô hình:
  - Điểm Health dự đoán đạt trung bình **0.966** ($\pm 0.026$). Có tới **99.9%** ảnh mưa thật bị mô hình phân lớp là **Healthy** (chỉ đúng 1/1 000 ảnh rơi vào mức $< 0.8$).
  - Trong khi đó, khi đo bằng proxy recall (đối chiếu YOLO11s với YOLO11x trên chính ảnh mưa thật), có tới **11%** số ảnh có proxy recall $< 0.70$. Tương quan giữa Health dự đoán và Proxy recall chỉ là $\text{Spearman} = -0.046$ (hoàn toàn không tương quan).
- **Nguyên nhân cốt lõi:**
  1. Trong ảnh thực tế của ACDC ([`reports/figures/real_rain_examples.png`](../figures/real_rain_examples.png)), mưa đường phố không phải là lớp sương mờ đồng nhất khắp ảnh. Thay vào đó, nó xuất hiện dưới dạng: **vệt cần gạt nước (wiper blade) chắn ngang một góc kính chắn gió**, đốm nước phản quang cục bộ, hoặc vũng nước phản chiếu dưới mặt đường.
  2. Toàn cảnh xung quanh vẫn có độ tương phản cao, tòa nhà và cây cối vẫn sắc nét. Do đó, 11 đặc trưng IQA toàn cục (như `laplacian_variance`, `contrast`, `entropy`) vẫn ghi nhận giá trị rất cao, khiến XGBoost kết luận camera hoàn toàn "khỏe mạnh".
  3. Đây là minh chứng rõ ràng nhất về sự thiếu hụt của suy biến nhân tạo toàn ảnh (Synthetic Uniform Corruption) khi đối mặt với điều kiện vận hành thực tế.
- **Minh chứng:** Ảnh minh hoạ [`reports/figures/real_rain_examples.png`](../figures/real_rain_examples.png) và log [`outputs/logs/health_model_run.log`](../../outputs/logs/health_model_run.log).

### Failure Case 2: Nhiễu lượng tử hoá nhãn khi số lượng vật thể ít ($n_{\text{gt}} = 3$)
- **Mô tả:** MAE của mô hình giảm mạnh theo số lượng object trong ảnh:
  - $3 - 4$ vật thể: $\text{MAE} = 0.190$.
  - $5 - 6$ vật thể: $\text{MAE} = 0.161$.
  - $\ge 10$ vật thể: $\text{MAE} = 0.115$.
- **Bản chất:** Ví dụ tại frame `GP020400_frame_000048`, ảnh chỉ có 3 vật thể nhỏ ở xa. Khi thêm Dark L1 hoặc Blur L1 nhẹ, YOLO11s trượt cả 3 vật thể $\to \text{Recall} = 0 \to \text{Health GT} = 0.0$ (Critical). Tuy nhiên, ảnh nhìn bằng mắt và qua 11 feature vẫn rất nét $\to$ XGBoost dự đoán $0.92 - 0.95$. Sự lệch pha này xuất phát từ bản chất rời rạc của phép đếm trên số lượng mẫu nhỏ chứ không phải mô hình học sai.

### Failure Case 3: Sự thất bại của các quy tắc Rule-based truyền thống
- Thử nghiệm mô hình Rule-based đo khoảng cách Euclidean tới phân bố ảnh sạch cho ra $R^2 = -0.2397$ và Spearman chỉ đạt $0.0664$.
- Lý do: Các đặc trưng IQA phản ứng phi tuyến và triệt tiêu lẫn nhau. Ví dụ: Nhiễu cảm biến (`noise`) làm tăng mạnh `laplacian_variance` (tạo cạnh giả), trong khi làm mờ (`blur`) lại làm giảm `laplacian_variance`. Do đó, không thể dùng bất kỳ ngưỡng tĩnh nào trên từng feature đơn lẻ để đánh giá độ tin cậy của camera.

### Failure Case 4: Vùng dao động của Detector ở các mức suy biến trung gian (Degraded class)
- Theo bảng nhầm lẫn ([`reports/figures/confusion_matrix.png`](../figures/confusion_matrix.png)), F1 của lớp Degraded chỉ đạt **0.46** (so với Healthy 0.84 và Critical 0.76).
- Có tới **743** ảnh Healthy bị đẩy nhầm sang Degraded. Đây là các trường hợp suy biến ở mức trung bình (Rain L2, Noise L2, Dark L2), nơi detector rơi vào vùng ranh giới "lúc nhận diện được, lúc bỏ sót", tạo ra độ lệch MAE cao nhất (~0.21) trên tập dữ liệu.

---

## 5. Engineering decision & trade-off

Bảng tổng hợp các quyết định kỹ thuật do tôi đưa ra và thống nhất cùng nhóm:

| Quyết định kỹ thuật | Cơ sở lý luận (Rationale) | Đánh đổi (Trade-off) |
|---|---|---|
| **1. Định nghĩa nhãn Health theo Perception Recall thay vì PSNR/SSIM** | Thí nghiệm tại Mục 4 cho thấy: Dark L1 làm sai lệch pixel rất lớn (mean $\|\Delta\| = 72$) nhưng detector vẫn đạt recall 0.93. Ngược lại, Rain L2 sai lệch pixel ít ($\|\Delta\| = 22$) nhưng mất tới 31% vật thể. Đánh giá chất lượng camera cho ADAS bắt buộc phải đo bằng năng lực nhận diện của AI. | Nhãn phụ thuộc vào mô hình detector cụ thể (YOLO11s). Nếu hệ thống ADAS đổi sang detector khác (RT-DETR hoặc Sparse R-CNN), cần tái tạo lại nhãn offline. |
| **2. Sử dụng YOLO11x làm Pseudo-GT Detector** | Tập ACDC trên Kaggle không cung cấp nhãn 2D Bounding Box chuẩn. Việc dùng YOLO11x (mô hình lớn nhất của series YOLO11) trên ảnh sạch cho phép tự động hoá gán nhãn cho 16 000 frame với độ tin cậy vượt trội so với YOLO11s. | Bỏ qua những vật thể mà ngay cả YOLO11x cũng không thể nhận diện được. Recall đo được là recall tương đối giữa 2 phiên bản detector. |
| **3. Lọc bỏ các ảnh có ít hơn 3 vật thể ($n_{\text{gt}} < 3$)** | Với $n_{\text{gt}} \in \{1, 2\}$, recall bị lượng tử hoá thô ($0\%, 50\%, 100\%$), tạo ra biến động nhãn giả tạo (F2) làm sai lệch quá trình hội tụ của XGBoost. | Chấp nhận loại bỏ ~25% số lượng ảnh gốc; tập huấn luyện ít xuất hiện các bối cảnh đường cao tốc vắng xe. |
| **4. Ánh xạ Camera Health $\to$ Trọng số thích ứng trong Sensor Fusion** | Cung cấp đầu ra thiết thực cho hệ thống tự hành: $w_i = h_i$. Khi $h_i < 0.5$, hệ thống lập tức loại camera đó khỏi late fusion và kích hoạt cảnh báo Take-Over / chuyển quyền ưu tiên cho LiDAR. | Cần cơ chế chuẩn hoá mềm ($\sum w_i = 1$) để đảm bảo không làm gián đoạn luồng dữ liệu của bộ theo dõi đa vật thể (Multi-Object Tracker). |
| **5. Giữ kiến trúc trích xuất 11 đặc trưng CPU thay vì End-to-End CNN** | Thời gian tính 11 IQA features chỉ mất ~0.12 s trên 1 core CPU, XGBoost suy luận $< 1\text{ ms}$. Phù hợp trực tiếp với chip nhúng ô tô (ECU) mà không đòi hỏi GPU chuyên dụng. | Mất khả năng nhận biết không gian cục bộ (spatial occlusion như cần gạt nước trong F1) do không sử dụng biểu diễn feature map sâu. |
| **6. Quyết định loại bỏ Grid 3x3 ở mô hình triển khai chính** | Mặc dù Grid 3x3 bổ sung 36 đặc trưng cục bộ, nhưng kết quả thực nghiệm cho thấy MAE trên test set bị giảm từ 0.1455 xuống 0.1548 (do overfit trên tập train chỉ có 3 930 mẫu tổng hợp toàn ảnh). | Mô hình gọn nhẹ hơn, nhưng nhóm ghi nhận việc cần bổ sung dữ liệu suy biến cục bộ trước khi tái kích hoạt Grid features. |

---

## 6. Kịch bản Demo & Ứng dụng thực tế

Hệ thống cung cấp kịch bản vận hành thực tế tại [`demo/app.py`](../../demo/app.py) và [`demo/adaptive_weight.py`](../../demo/adaptive_weight.py), tích hợp vào kiến trúc ADAS đa cảm biến:

```
[Camera Image] ---> [11 IQA Features] ---> [XGBoost Regressor] ---> Health h in [0, 1]
                                                                            │
      ┌─────────────────────────────────────────────────────────────────────┘
      ▼
[Health Classification & ADAS Action Policy]:
  ├── Healthy  (h >= 0.8): weight = h; Normal autonomous cruise.
  ├── Degraded (0.5 <= h < 0.8): weight = 0.5 * h; Warning! Re-weighting fusion; LiDAR scan boost.
  └── Critical (h < 0.5): weight = 0.0; EMERGENCY! Isolate camera, Fallback to LiDAR/Radar!
```

### Minh chứng mô phỏng Multi-Camera Fusion ([`outputs/logs/demo_predict.log`](../../outputs/logs/demo_predict.log)):
Khi xe di chuyển trong bối cảnh mưa lớn làm camera trước bị che khuất (`h = 0.202`), trong khi 3 camera xung quanh vẫn nhìn tốt:
- **CAM_FRONT:** Health = 0.202 $\to$ Trọng số chuẩn hoá hạ còn **0.0684** (giảm 73% độ tin cậy).
- **CAM_LEFT / RIGHT / BACK:** Health $\ge 0.880 \to$ Trọng số chuẩn hoá tự động tăng lên **~0.30 - 0.32**.
- **ADAS Controller Action:** Phát tín hiệu: *"WARNING: Front camera degraded (health=0.20). Shifting perception priority to LiDAR and Radar front clusters."*

---

## 7. Bằng chứng chạy & Tái lập kết quả

Toàn bộ các phân hệ do tôi phụ trách đều có thể chạy lại và kiểm chứng thông qua các lệnh terminal chuẩn:

```bash
# 1. Kích hoạt môi trường và kiểm tra mã nguồn
python -m perception.detector
python -m perception.evaluate

# 2. Chạy phân tích tương quan giữa Health dự đoán và Detection Recall
python -m analysis.correlation

# 3. Trực quan hoá biểu đồ phân tán và suy giảm recall
python -m analysis.visualization

# 4. Thử nghiệm cơ chế điều phối trọng số Multi-Camera Adaptive Fusion
python -m demo.adaptive_weight

# 5. Chạy ứng dụng Demo ADAS (hỗ trợ cả giao diện CLI và Streamlit)
python -m demo.app
# Hoặc khởi chạy giao diện web:
# streamlit run demo/app.py

# 6. Kiểm tra suy luận trên ảnh mẫu thực nghiệm
python -m health_model.predict outputs/results/demo_predict.csv
```

### Trích xuất Log thực thi kiểm chứng trên hệ thống:

```text
=== PER-CONDITION DETECTOR EVALUATION (YOLO11s vs YOLO11x Pseudo-GT) ===
Degradation        | Level | Images  | Mean Recall  | Mean Precision | Total GT | Total TP
-------------------------------------------------------------------------------------
blur               | 1     | 1000    | 0.8804       | 0.6070         | 5600     | 4913    
blur               | 2     | 1000    | 0.7765       | 0.6498         | 5600     | 4311    
blur               | 3     | 1000    | 0.5805       | 0.6789         | 5600     | 3160    
blur               | 4     | 1000    | 0.2130       | 0.4879         | 5600     | 1184    
dark               | 1     | 1000    | 0.8721       | 0.5774         | 5600     | 4871    
dark               | 2     | 1000    | 0.6720       | 0.6375         | 5600     | 3702    
noise              | 1     | 1000    | 0.8542       | 0.5710         | 5600     | 4771    
noise              | 2     | 1000    | 0.6391       | 0.6274         | 5600     | 3555    
noise              | 3     | 1000    | 0.2629       | 0.5066         | 5600     | 1518    
none               | 0     | 1000    | 0.9285       | 0.5479         | 5600     | 5168    
overexposure       | 1     | 1000    | 0.9170       | 0.5464         | 5600     | 5123    
overexposure       | 2     | 1000    | 0.8341       | 0.5373         | 5600     | 4694    
rain               | 1     | 1000    | 0.8774       | 0.5712         | 5600     | 4915    
rain               | 2     | 1000    | 0.6297       | 0.6013         | 5600     | 3556    
rain               | 3     | 1000    | 0.2019       | 0.4214         | 5600     | 1082    

=== CORRELATION ANALYSIS: CAMERA HEALTH vs DETECTION RECALL ===
Condition/Subset     | N      | Pearson (Pred, Recall)   | Spearman (Pred, Recall)   | Spearman (Pred, GT) 
---------------------------------------------------------------------------------------------------------
ALL (Test set)       | 6330   | 0.7715                   | 0.6870                    | 0.7175              
rain                 | 1266   | 0.7816                   | 0.7384                    | 0.7540              
blur                 | 1688   | 0.7608                   | 0.7223                    | 0.7417              
noise                | 1266   | 0.7286                   | 0.6990                    | 0.7139              
dark                 | 844    | 0.4399                   | 0.4133                    | 0.4332              
overexposure         | 844    | 0.1972                   | 0.1453                    | 0.2553              
REAL RAIN (OOD)      | 932    | -0.0552                  | -0.0302                   | N/A (No GT)         

>> Nhan xet Real Rain Failure Case: Spearman(pred, proxy) ~ 0 phan anh ro domain gap synthetic -> real.
```

---

## 8. Kết luận & Hướng phát triển cá nhân

Phân hệ **ADAS Validation & Adaptive Weighting** đã hoàn thành mục tiêu:
1. Xác lập phương pháp đo lường "sức khoẻ" cảm biến gắn liền với hiệu năng của hệ thống nhận thức ADAS thực tế (thay vì các chỉ số thị giác nhân tạo thuần túy).
2. Xây dựng cầu nối trực tiếp giữa điểm số dự đoán của mô hình học máy và cơ chế điều phối trọng số cảm biến trong xe tự hành, đóng góp giải pháp thiết thực cho bài toán an toàn chức năng (Functional Safety - ISO 26262 & SOTIF ISO 21448).
3. Chỉ ra và định lượng rõ ràng bài toán Domain Gap trên ảnh thời tiết thực tế, mở ra hướng nghiên cứu tiếp theo: bổ sung tập dữ liệu mô phỏng hư hại cục bộ (wiper occlusion, dirt/mud splatter) kết hợp cơ chế chú ý không gian (Spatial Attention) để hoàn thiện hệ thống giám sát sức khoẻ camera toàn diện.
