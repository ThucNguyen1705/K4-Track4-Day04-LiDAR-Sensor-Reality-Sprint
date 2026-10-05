# Báo cáo cá nhân — Ngô Xuân Hoàng (2A202602597)

**Vai trò:** Dataset + Data Pipeline · **Repo nhóm:** [https://github.com/ThucNguyen1705/K4-Track4-Day04-LiDAR-Sensor-Reality-Sprint](https://github.com/ThucNguyen1705/K4-Track4-Day04-LiDAR-Sensor-Reality-Sprint) · **Báo cáo nhóm:** [../REPORT.md](../REPORT.md)

---

## 1. Phần việc của tôi

Tôi chịu trách nhiệm phân hệ **Dataset & Data Pipeline**, phụ trách thiết kế, thu thập, tiền xử lý, sinh các dạng suy biến cảm biến (Sensor Degradation Engine), phân chia tập dữ liệu chống rò rỉ (Zero-Leakage Split), và ban hành chuẩn dữ liệu chung (Data Contract). Các thư mục và file mã nguồn do tôi phụ trách chính bao gồm:

| Module / Script | File cụ thể | Vai trò và nhiệm vụ chính |
|---|---|---|
| **Data Contract & Metadata** | [`data/build_metadata.py`](../../data/build_metadata.py) | Xây dựng pipeline tổng hợp toàn bộ 16 000 ảnh (clean + 14 biến thể suy biến + ảnh mưa thật) xuất ra [`dataset/metadata.csv`](../../dataset/metadata.csv) theo đúng schema chuẩn của nhóm. |
| **Degradation Engine** | [`data/corruption/corrupt.py`](../../data/corruption/corrupt.py) | Cài đặt 14 biến thể suy biến cảm biến quang học với nguyên tắc **Pixel-Aligned**: Rain (L1–3), Blur (L1–4), Gaussian Noise (L1–3), Dark (L1–2), Overexposure (L1–2). |
| **Data Downloader** | [`data/download/download.py`](../../data/download/download.py) | Quản lý việc tải và cấu trúc hoá tập dữ liệu ACDC Rain (`rgb_anon`) từ Kaggle ([`njayadithya/rgb-anon-trainvaltest`](https://www.kaggle.com/datasets/njayadithya/rgb-anon-trainvaltest)). |
| **Image Preprocessing** | [`data/preprocessing/preprocess.py`](../../data/preprocessing/preprocess.py) | Chuẩn hoá toàn bộ ảnh về độ phân giải $1280 \times 720$ (16:9) sử dụng thuật toán nội suy `cv2.INTER_AREA` để triệt tiêu hiện tượng răng cưa (aliasing). |
| **Sequence Split Strategy** | [`data/preprocessing/split.py`](../../data/preprocessing/split.py) | Thiết kế thuật toán phân chia Train / Val / Test theo `sequence_id` của ACDC, đảm bảo không rò rỉ cảnh (zero sequence leakage). |
| **Kaggle GPU Batch Processing** | [`kaggle/pipeline.py`](../../kaggle/pipeline.py) | Cài đặt các hàm xử lý suy biến song song đa nhân (`synth_rain`, `degrade`, `worker`) để xử lý trơn tru 16 000 ảnh trên Kaggle GPU T4 trong 13 phút. |
| **Ground Truth Diff & Zoom Analysis** | [`reports/make_figures.py`](../../reports/make_figures.py) | Phối hợp phân tích sai khác pixel tuyệt đối ($|\Delta|$) so với độ suy giảm perception, trực quan hoá lưới 15 biến thể ([`examples_*.png`](../figures/)), bản đồ sai khác ([`diff_*.png`](../figures/)), và phóng to chi tiết vật thể bị mất ([`zoom_*.png`](../figures/)). |

---

## 2. Nguồn tham khảo (paper / repo)

1. **ACDC Dataset:** Christos Sakaridis, Dengxin Dai, Luc Van Gool. *ACDC: The Adverse Conditions Dataset with Correspondences.* IEEE/CVF International Conference on Computer Vision (ICCV), 2021.  
   - Cung cấp tập dữ liệu lái xe thực tế với các chuỗi ảnh trong điều kiện mưa thật kèm ảnh tham chiếu trời quang cùng hành trình (`*_ref`). Đây là nền tảng để nhóm xây dựng các cặp ảnh đối chứng.
2. **Robustness & Synthetic Corruptions Benchmark:** Dan Hendrycks, Thomas Dietterich. *Benchmarking Neural Network Robustness to Common Corruptions and Perturbations* (ImageNet-C). ICLR 2019.  
   - Cơ sở lý thuyết nền tảng để mô hình hoá toán học các dạng suy biến cảm biến vật lý phổ biến: Gaussian blur (mất nét ống kính), Gaussian noise (nhiễu nhiệt/ISO cảm biến), Dark/Overexposure (giới hạn dải động - Dynamic Range).
3. **Mô hình hoá Mưa và Tán xạ Khí quyển (Physical Rain Streaks & Veiling):**  
   - K. Garg, S. K. Nayar. *Vision and Rain.* International Journal of Computer Vision (IJCV), 2007.  
   - W. Yang et al. *Deep Joint Rain Detection and Removal.* IEEE CVPR, 2017.  
   - Áp dụng nguyên lý kết hợp 3 thành phần trong mô phỏng mưa: vệt mưa chuyển động góc nghiêng ($\pm 15^\circ$), lớp sương mờ tán xạ ánh sáng (Atmospheric Veiling / Haze), và hiệu ứng nhòe do giọt nước bám kính chắn gió (Droplet Blur).
4. **Ultralytics YOLO11:** Glenn Jocher et al. *Ultralytics YOLO11*, 2024. Repo: [github.com/ultralytics/ultralytics](https://github.com/ultralytics/ultralytics).  
   - Quy chuẩn kích thước đầu vào và kiểm soát bounding box vật thể cho 6 lớp ADAS trọng yếu (person, bicycle, car, motorcycle, bus, truck).
5. **Data Contract & Data Engineering trong ML Systems:**  
   - E. Breck et al. *Data Validation for Machine Learning.* SysML, 2019.  
   - Tiêu chuẩn an toàn chức năng và SOTIF trong xe tự hành (ISO 26262 & ISO 21448) về quản lý rủi ro suy thoái cảm biến quang học.

---

## 3. Cách nhóm đo metric

### 3.1. Thiết kế dữ liệu và định nghĩa nhãn Health ADAS-Centric
Trong các bài toán xử lý ảnh truyền thống, chất lượng ảnh thường được đo bằng các chỉ số thị giác con người như PSNR (Peak Signal-to-Noise Ratio) hoặc SSIM (Structural Similarity Index). Tuy nhiên, trong hệ thống tự hành ADAS, câu hỏi cốt lõi là: **"Camera này còn đủ tin cậy để mạng nơ-ron nhận diện vật thể hay không?"**

Do đó, nhóm tôi thống nhất định nghĩa nhãn **Camera Health** dựa trên năng lực nhận thức thực tế:

$$\text{Health} = \text{clip}\left(\frac{\text{Recall}_{\text{YOLO11s}}(\text{ảnh degraded})}{\text{Recall}_{\text{YOLO11s}}(\text{ảnh clean cùng cảnh})}, 0.0, 1.0\right)$$

Để tính toán được công thức này một cách chuẩn xác:
1. **Thuộc tính Pixel-Aligned bắt buộc:** Khi tôi tạo 14 biến thể suy biến (rain, blur, noise, dark, overexposure), toàn bộ toạ độ hình học $(x, y)$ của mọi vật thể trong ảnh được giữ **nguyên vẹn 100%** so với ảnh sạch gốc. Nhờ đó, việc đối chiếu matching Bounding Box ($\text{IoU} \ge 0.5$ cùng class) giữa detector trên ảnh suy biến và Pseudo-GT trên ảnh sạch diễn ra tuyệt đối khách quan, không bị sai lệch do biến đổi toạ độ.
2. **Chiến lược phân chia Zero-Leakage:** 1 000 ảnh sạch gốc `*_ref` của ACDC được phân chia theo `sequence_id`:
   - **Train:** 400 ảnh gốc $\to$ sinh 6 000 ảnh biến thể.
   - **Val:** 100 ảnh gốc $\to$ sinh 1 500 ảnh biến thể.
   - **Test:** 500 ảnh gốc $\to$ sinh 7 500 ảnh biến thể.
   - Các biến thể suy biến kế thừa nguyên vẹn `split` của ảnh mẹ (`parent_id`).
3. **Lọc nhiễu lượng tử hoá:** Loại bỏ các ảnh có số lượng vật thể trong Pseudo-GT $n_{\text{gt}} < 3$ hoặc $\text{Recall}_{\text{clean}} = 0$. Sau bước lọc, tập dữ liệu còn lại **11 280 nhãn hợp lệ**: Train 3 930 · Val 1 020 · Test 6 330.

```
+---------------------------------------------------------------------------------------------------+
|                               PHÂN BỐ TẬP DỮ LIỆU CAMERA HEALTH                                   |
+---------------------+-------------------+-------------------+------------------+------------------+
| Tập dữ liệu         | Số ảnh sạch gốc   | Tổng biến thể     | Sau lọc n_gt >= 3| Tỉ lệ phân bổ    |
+---------------------+-------------------+-------------------+------------------+------------------+
| Train               | 400 ảnh           | 6 000 ảnh         | 3 930 nhãn       | 34.8%            |
| Validation          | 100 ảnh           | 1 500 ảnh         | 1 020 nhãn       | 9.0%             |
| Test (độc lập)      | 500 ảnh           | 7 500 ảnh         | 6 330 nhãn       | 56.2%            |
| Real Rain (OOD)     | 1 000 ảnh         | 1 000 ảnh         | 932 kiểm chứng   | Đánh giá OOD     |
| TỔNG CỘNG           | 2 000 ảnh         | 16 000 ảnh        | 11 280 nhãn GT   | 100%             |
+---------------------+-------------------+-------------------+------------------+------------------+
```

### 3.2. Đối chiếu Sai khác cấp độ Pixel ($|\Delta|$) vs Suy giảm Perception
Một đóng góp thực nghiệm quan trọng từ pipeline dữ liệu của tôi là phân tích định lượng mối quan hệ giữa **độ sai lệch pixel trung bình** $\text{mean } |\Delta| = \frac{1}{HW}\sum |I_{\text{degraded}} - I_{\text{clean}}|$ và **độ suy giảm nhận diện của AI**:

```
+--------------------------------------------------------------------------------------------------------+
|                      ĐỐI CHIẾU SAI KHÁC PIXEL (|Δ|) VS NĂNG LỰC NHẬN DIỆN ADAS                        |
+-------------------+-------+---------------+-----------------+------------------+-----------------------+
| Loại suy biến     | Cấp độ| Sai khác |Δ|  | True Mean Health| Recall detector  | Nhận xét bản chất     |
+-------------------+-------+---------------+-----------------+------------------+-----------------------+
| Clean (none)      | L0    | 0.0           | 1.000           | 0.928            | Trạng thái lý tưởng   |
| Dark              | L1    | 72.1 mức xám  | 0.933           | 0.872            | Sai khác pixel cực lớn|
| Dark              | L2    | 91.4 mức xám  | 0.731           | 0.672            | nhưng AI vẫn nhận tốt |
| Overexposure      | L1    | 28.5 mức xám  | 0.982           | 0.917            | Cháy sáng nền trời,   |
| Overexposure      | L2    | 45.2 mức xám  | 0.910           | 0.834            | xe/người vẫn rõ nét   |
| Rain              | L1    | 12.3 mức xám  | 0.946           | 0.877            | Vệt mưa nhẹ chưa hại  |
| Rain              | L2    | 22.4 mức xám  | 0.690           | 0.630            | Mất 31% vật thể xa    |
| Rain              | L3    | 38.6 mức xám  | 0.219           | 0.202            | Mất gần hết object    |
| Blur              | L1    | 8.4 mức xám   | 0.945           | 0.880            | Nhòe cạnh nhẹ         |
| Blur              | L2    | 14.1 mức xám  | 0.837           | 0.776            | Mất chi tiết nhỏ      |
| Blur              | L3    | 20.7 mức xám  | 0.623           | 0.581            | Mất vật thể xa        |
| Blur              | L4    | 28.9 mức xám  | 0.254           | 0.213            | Mất cả xe lớn ở gần   |
| Noise             | L1    | 6.2 mức xám   | 0.918           | 0.854            | Nhiễu hạt nhẹ         |
| Noise             | L2    | 15.6 mức xám  | 0.703           | 0.639            | Phá vỡ kết cấu biên   |
| Noise             | L3    | 24.3 mức xám  | 0.315           | 0.263            | Sai khác bé nhưng mất |
|                   |       |               |                 |                  | tới 68.5% vật thể!    |
+-------------------+-------+---------------+-----------------+------------------+-----------------------+
```

*Kết luận mang tính bước ngoặt từ dữ liệu:* **Độ biến thiên pixel không phản ánh sự suy giảm của hệ thống nhận thức**. Điển hình là Dark L1 làm lệch pixel tới 72 mức xám nhưng detector vẫn đạt recall 87% (Health 0.933). Ngược lại, Noise L3 chỉ lệch 24 mức xám nhưng phá nát kết cấu cục bộ, khiến detector mất tới gần 70% số vật thể (Health chỉ còn 0.315). Điều này chứng minh rằng việc đánh giá chất lượng camera bằng PSNR/SSIM là sai lầm nguy hiểm trong ADAS, và khẳng định tính đúng đắn của việc xây dựng nhãn theo perception recall.

### 3.3. Kết quả đo lường Benchmark trên Test Set (6 330 ảnh)
Từ tập dữ liệu và nhãn do tôi trích xuất, mô hình XGBoost và các mô hình cơ sở đạt kết quả trên Test Set như sau ([`outputs/results/results.csv`](../../outputs/results/results.csv)):

| Mô hình | Số Feature | MAE ↓ | RMSE ↓ | $R^2$ ↑ | Pearson ↑ | Spearman ↑ | Accuracy | Macro-F1 |
|---|---|---|---|---|---|---|---|---|
| **Rule-based (Khoảng cách Euclidean tới clean)** | 11 | 0.2543 | 0.3600 | -0.2397 | 0.0154 | 0.0664 | 0.5226 | 0.3078 |
| **Linear Regression** | 11 | 0.1785 | 0.2299 | 0.4943 | 0.7118 | 0.6745 | 0.6264 | 0.6068 |
| **Random Forest** | 11 | **0.1444** | 0.2017 | 0.6108 | 0.7936 | **0.7467** | 0.7226 | 0.6777 |
| **LightGBM** | 11 | 0.1477 | 0.2057 | 0.5952 | 0.7877 | 0.7349 | 0.7172 | 0.6643 |
| **XGBoost (Mô hình triển khai)** | 11 | **0.1455** | **0.1991** | **0.6207** | **0.7964** | **0.7460** | **0.7329** | **0.6857** |
| **XGBoost + Grid 3x3** | 47 | 0.1548 | 0.2079 | 0.5866 | 0.7870 | 0.7379 | 0.7118 | 0.6713 |

---

## 4. Failure cases

Dưới góc nhìn của một kỹ sư phụ trách **Dataset & Data Pipeline**, tôi phân tích 4 Failure Cases chính của hệ thống:

```
+---------------------------------------------------------------------------------------------------+
|                                      BẢN ĐỒ PHÂN TÍCH FAILURE CASES                                |
+---------------------------------------------------------------------------------------------------+
|  F1: Domain Gap giữa Rain Tổng Hợp (Synthetic) và Mưa Thực Tế (Real Rain ACDC)                    |
|      - Dữ liệu: Model chấm mean 0.966 (99.9% Healthy), proxy recall < 0.70 ở 11% ảnh               |
|      - Bản chất: Synthetic rain phủ đều; Real rain có cần gạt nước (wiper) che khuất cục bộ      |
+---------------------------------------------------------------------------------------------------+
|  F2: Lượng tử hoá nhãn rời rạc khi ảnh có ít vật thể (n_gt = 3 - 4)                               |
|      - Dữ liệu: MAE nhảy vọt lên 0.190 (so với 0.115 ở ảnh có >= 10 vật thể)                      |
|      - Bản chất: Bước nhảy gián đoạn (0, 0.33, 0.67, 1.0) gây nhiễu gradient cho mô hình hồi quy |
+---------------------------------------------------------------------------------------------------+
|  F3: Hiện tượng Đối kháng Feature (Feature Antagonism) khiến Rule-based thất bại                  |
|      - Dữ liệu: Rule-based đạt R² = -0.2397, Spearman = 0.0664                                    |
|      - Bản chất: Noise làm Laplacian tăng vọt, Blur làm giảm sụt; không thể phân tách tuyến tính  |
+---------------------------------------------------------------------------------------------------+
|  F4: Độ biến thiên bất định ở các mức suy biến trung gian (Degraded class)                        |
|      - Dữ liệu: Lớp Degraded có F1 thấp nhất (0.46), MAE chạm đỉnh ~0.21                          |
|      - Bản chất: Vùng ranh giới nhấp nháy của detector (Rain L2, Noise L2, Blur L3, Dark L2)     |
+---------------------------------------------------------------------------------------------------+
```

### Failure Case 1: Lỗ hổng Domain Gap giữa Mưa nhân tạo và Mưa thực tế
- **Hiện tượng thực nghiệm:**
  - Khi đưa 1 000 ảnh mưa thật của tập ACDC (`outputs/kaggle/real_rain_proxy.csv`) qua mô hình đã huấn luyện, điểm Health dự đoán đạt trung bình **0.966** ($\pm 0.026$). Có tới **99.9%** ảnh mưa thật bị mô hình xếp vào lớp **Healthy** (chỉ đúng duy nhất 1 ảnh rơi vào mức $< 0.8$).
  - Tuy nhiên, khi đo bằng proxy recall (đối chiếu YOLO11s với YOLO11x trên chính ảnh mưa thật), có tới **11%** số ảnh có proxy recall $< 0.70$. Hệ số tương quan Spearman giữa Health dự đoán và Proxy recall thực tế rơi về **$-0.046$** (hoàn toàn mất liên kết).
- **Phân tích nguyên nhân gốc rễ từ Data Pipeline:**
  1. Trong hàm `synth_rain`, mô hình toán học giả định mưa là một hiện tượng quang học **đồng nhất trên toàn khung hình** (uniform spatial degradation): tạo $N$ đường kẻ mảnh ngẫu nhiên phủ khắp chiều rộng $W$ và chiều cao $H$, kết hợp một lớp sương mờ Gaussian đồng nhất.
  2. Trong thực tế thu thập từ ACDC ([`reports/figures/real_rain_examples.png`](../figures/real_rain_examples.png)), mưa đường phố tạo ra các hình thái vật lý phi đồng nhất:
     - **Vệt cần gạt nước (wiper blade occlusion):** Lưỡi gạt nước tạo thành một dải bán nguyệt chắn ngang tầm nhìn camera, cuốn theo bụi bẩn và vệt nước đậm đặc.
     - **Đốm nước bám ống kính (water droplets on lens):** Các đốm khúc xạ ánh sáng cục bộ.
     - **Vũng nước mặt đường phản chiếu đèn xe (wet road reflections / glare).**
  3. Xung quanh vệt cần gạt nước, khung cảnh nhà cửa và cây cối vẫn có độ nét cao, tương phản mạnh. Do đó, các đặc trưng thống kê toàn cảnh như `laplacian_variance` hay `entropy` vẫn giữ giá trị cao, khiến mô hình kết luận camera hoàn toàn khỏe mạnh trong khi một góc quan trọng đã bị che mù.
- **Biện pháp khắc phục về mặt dữ liệu:**
  - Cần nâng cấp pipeline sinh dữ liệu suy biến: bổ sung mặt nạ suy biến cục bộ (Local Occlusion Masks) mô phỏng chuyển động của cần gạt nước và giọt nước bám kính. Khi dữ liệu có suy biến cục bộ, hệ thống feature lưới 3×3 (`features/spatial.py`) sẽ phát huy tối đa sức mạnh.

### Failure Case 2: Nhiễu lượng tử hoá nhãn ở các khung hình ít vật thể ($n_{\text{gt}} < 3$)
- **Hiện tượng thực nghiệm:**
  - MAE của mô hình giảm tỷ lệ nghịch với số lượng vật thể trong ảnh:
    - Nhóm $3 - 4$ vật thể: $\text{MAE} = 0.190$.
    - Nhóm $5 - 6$ vật thể: $\text{MAE} = 0.161$.
    - Nhóm $\ge 10$ vật thể: $\text{MAE} = 0.115$.
- **Phân tích bản chất:**
  - Điển hình tại frame `GP020400_frame_000048`: ảnh gốc chỉ có 3 vật thể kích thước nhỏ ở phía xa chân trời. Khi thêm Dark L1 hoặc Blur L1 nhẹ, YOLO11s trượt cả 3 vật thể này $\to \text{Recall} = 0 \to \text{Health GT} = 0.0$ (phân lớp Critical).
  - Tuy nhiên, về mặt thị giác và các chỉ số IQA, ảnh vẫn sáng rõ và rất nét $\to$ XGBoost dự đoán điểm $0.92 - 0.95$. Sự chênh lệch này không phải do mô hình học kém, mà do nhãn bị lượng tử hoá quá thô (bước nhảy $0.33$ hoặc $1.0$).
- **Xử lý từ phía Data Pipeline:**
  - Tôi đã thiết lập bộ lọc tự động trong pipeline: loại bỏ toàn bộ các frame có $n_{\text{gt}} < 3$. Mặc dù việc này làm giảm 25% lượng frame sử dụng, nó đã loại bỏ hoàn toàn các điểm kỳ dị (outliers) gây nhiễu gradient cho quá trình tối ưu của cây tăng cường (XGBoost).

### Failure Case 3: Hiện tượng Đối kháng Feature (Feature Antagonism)
- **Hiện tượng:**
  - Mô hình Rule-based dựa trên khoảng cách Euclidean đến phân bố ảnh sạch thất bại nặng nề: $R^2 = -0.2397$, Spearman chỉ đạt $0.0664$.
- **Phân tích cơ chế suy biến dữ liệu:**
  - Mỗi loại suy biến tác động lên các đặc trưng hình ảnh theo những hướng hoàn toàn đối nghịch:
    - `blur`: làm mờ triệt tiêu tần số cao $\to$ `laplacian_variance` giảm từ $-4$ đến $-9$ lần (theo thang $\log_2$).
    - `noise`: nhiễu hạt Gaussian tần số cao $\to$ `laplacian_variance` và `noise_estimate` lại tăng vọt gấp 3–4 lần.
    - `rain L3`: các vệt mưa sọc trắng lại làm tăng giả tạo mật độ biên cạnh `edge_density`.
  - Do đó, không tồn tại bất kỳ một siêu phẳng tuyến tính hay ngưỡng cắt đơn biến (univariate threshold) nào có thể phân biệt được trạng thái camera. Cần phải có các bộ phân tách phi tuyến tính đa tầng như XGBoost để nhận diện "dấu vân tay" kết hợp của 11 đặc trưng.

### Failure Case 4: Độ biến thiên bất định ở các mức suy biến trung gian (Degraded Class)
- **Hiện tượng:**
  - Trong bảng kết quả theo từng cấp độ ([`outputs/results/results_by_degradation.csv`](../../outputs/results/results_by_degradation.csv)), sai số MAE phân bổ theo hình chuông: cực thấp ở hai đầu (Clean: 0.0295, Overexposure L1: 0.0542) nhưng tăng vọt ở mức trung bình: Rain L2 (0.2132), Noise L2 (0.2123), Blur L3 (0.2167), Dark L2 (0.2031).
  - Lớp Degraded có chỉ số F1 thấp nhất (0.46), và có 743 ảnh Healthy bị phân lớp nhầm thành Degraded.
- **Phân tích:**
  - Mức suy biến trung gian là vùng ranh giới nhạy cảm (borderline uncertainty) của mạng nơ-ron nhận diện vật thể. Tại đây, chỉ cần một biến động nhỏ về ánh sáng nền hoặc góc quay xe, detector có thể lúc nhận diện được lúc bỏ sót, tạo ra phương sai nhãn lớn nhất trong toàn bộ tập dữ liệu.

---

## 5. Engineering decisions & trade-offs

Dưới đây là các quyết định kỹ thuật then chốt mà tôi (Ngô Xuân Hoàng) đã đề xuất và triển khai trong phân hệ dữ liệu:

| Quyết định kỹ thuật | Cơ sở lý luận (Rationale) | Đánh đổi (Trade-off) |
|---|---|---|
| **1. Chuẩn hoá độ phân giải cố định $1280 \times 720$ (16:9)** | Tập dữ liệu ACDC gốc có độ phân giải $1920 \times 1080$. Việc resize về $1280 \times 720$ bằng thuật toán `cv2.INTER_AREA` giúp giảm **55.6%** dung lượng bộ nhớ và thời gian tính toán trích xuất đặc trưng; đồng thời tỷ lệ 16:9 tương thích hoàn hảo với bộ đệm của camera ô tô hiện đại và kiến trúc đầu vào của YOLO11. | Mất đi một số chi tiết cực nhỏ của các phương tiện ở khoảng cách rất xa ($> 150\text{ m}$). Các đặc trưng nhạy với độ phân giải như `laplacian_variance` bị thu nhỏ khoảng $\times 2.3$ lần so với ảnh Full HD (đòi hỏi data contract phải khoá chặt độ phân giải). |
| **2. Sinh 14 biến thể suy biến dạng Pixel-Aligned** | Để đo lường chính xác tỷ lệ Recall mà không bị nhiễu do sai lệch góc nhìn hay chuyển động, toàn bộ các suy biến (mưa, mờ, nhiễu, tối, cháy sáng) được tạo trực tiếp trên ảnh gốc sao cho toạ độ bounding box không đổi. | Phải sử dụng suy biến nhân tạo (Synthetic Degradation). Điều này dẫn đến khoảng cách miền (Domain Gap - F1) khi đối mặt với các dạng thời tiết thực tế phức tạp ngoài phân phối. |
| **3. Phân chia Train/Val/Test theo `sequence_id` (Scene-level)** | Nếu chia ảnh ngẫu nhiên (random frame split), các frame liên tiếp trong cùng một video clip hành trình (cùng toà nhà, cùng hàng cây) sẽ xuất hiện ở cả train và test, gây hiện tượng học vẹt (Data Leakage) nghiêm trọng. Chia theo sequence đảm bảo kiểm thử mô hình trên các cung đường hoàn toàn mới lạ. | Kích thước các chuỗi trong ACDC không đồng đều. Tập train chỉ có 400 ảnh gốc (3 930 mẫu sau lọc) trong khi tập test có tới 500 ảnh gốc (6 330 mẫu sau lọc). Tập huấn luyện tương đối nhỏ đòi hỏi mô hình phải có khả năng khái quát cao và kiểm soát overfit chặt chẽ. |
| **4. Thiết lập Data Contract nghiêm ngặt với Primary Key `image_id` và `parent_id`** | Quy định schema nhất quán giữa 5 thành viên nhóm: mọi file trung gian (`metadata.csv`, `features.csv`, `per_image_metrics.csv`, `health_labels.csv`, `predictions.csv`) đều liên kết 1-1 qua `image_id`. Khóa `parent_id` cho phép truy vết trực tiếp về ảnh sạch đối chứng để tính tỷ lệ health. | Bất kỳ sự thay đổi nào về định dạng tên file hay kiểu dữ liệu đều đòi hỏi cả nhóm phải đồng bộ lại code, không được tự ý sửa đổi cục bộ. |
| **5. Cố định Pseudo-Random Seed (`zlib.crc32(key)`) trong sinh nhiễu** | Trong hàm `worker` của `kaggle/pipeline.py`, seed ngẫu nhiên để sinh hạt mưa và nhiễu Gaussian được tính toán tất định từ mã CRC32 của tên frame. Điều này đảm bảo tính tái lập (reproducibility) 100% của toàn bộ 16 000 ảnh dù chạy trên Kaggle GPU hay máy local. | Không tạo ra sự biến thiên vô hạn của dữ liệu trong quá trình huấn luyện như phương pháp online augmentation thông thường. |
| **6. Lọc bỏ các khung hình có $n_{\text{gt}} < 3$** | Khung hình có quá ít vật thể khiến nhãn recall bị rời rạc thô thiển (F2). Việc lọc bỏ giúp ổn định hàm mất mát (loss function) của mô hình hồi quy. | Mất khoảng 25% lượng frame dữ liệu; mô hình ít được tiếp xúc với các bối cảnh đường cao tốc ngoại ô ban đêm hoàn toàn vắng bóng xe cộ. |

---

## 6. Kịch bản Demo, Ứng dụng ADAS & Minh chứng Pipeline

Pipeline dữ liệu không chỉ phục vụ huấn luyện offline mà còn đóng vai trò là kiến trúc nền tảng cho hệ thống giám sát sức khoẻ cảm biến theo thời gian thực (Camera Health Monitoring Runtime).

```
               LUỒNG VẬN HÀNH THỜI GIAN THỰC TRONG ADAS
               
   [Camera Cảm Biến Ô Tô] (1280x720 @ 30fps)
              │
              ▼
   [Preprocess: Chuẩn hoá kích thước & kiểm tra schema]
              │
              ▼
   [Trích xuất 11 Đặc trưng IQA siêu nhẹ trên CPU (~0.12 s)]
              │
              ▼
   [XGBoost Inference (< 1 ms)] ──► Health Score h ∈ [0, 1]
                                          │
        ┌─────────────────────────────────┴─────────────────────────────────┐
        ▼                                 ▼                                 ▼
   Healthy (h >= 0.8)             Degraded (0.5 <= h < 0.8)          Critical (h < 0.5)
   - Trọng số cam w = h           - Trọng số cam w = 0.5*h           - Trọng số cam w = 0.0
   - Hệ thống tự hành             - Cảnh báo suy giảm chất lượng     - BẬT CẢNH BÁO NGUY HIỂM!
     hoạt động bình thường        - Tăng tần số quét LiDAR           - Cách ly camera lập tức
                                  - Yêu cầu tài xế chú ý             - Fallback toàn bộ sang LiDAR/Radar
```

### Minh chứng chạy thực nghiệm trên các biến thể cùng một cảnh
Khi chạy kiểm thử trên chuỗi biến thể của cảnh `GOPR0572_frame_000255_ref` ([`outputs/logs/demo_predict.log`](../../outputs/logs/demo_predict.log)), hệ thống đưa ra phản ứng chính xác:

| File ảnh đầu vào | Health dự đoán | Phân lớp ADAS | Trọng số Fusion | Hành vi điều khiển xe |
|---|---|---|---|---|
| `GOPR0572_frame_000255_ref.jpg` (Clean) | **0.963** | **Healthy** | **0.963** | Vận hành bình thường (Normal Cruise) |
| `GOPR0572_frame_000255_rain1.jpg` (Rain L1) | **0.885** | **Healthy** | **0.885** | Giữ làn tự động, duy trì tốc độ |
| `GOPR0572_frame_000255_rain2.jpg` (Rain L2) | **0.674** | **Degraded** | **0.674** | Cảnh báo mưa; tăng độ tin cậy cụm LiDAR trước |
| `GOPR0572_frame_000255_rain3.jpg` (Rain L3) | **0.202** | **Critical** | **0.202** | Ngắt camera trước; chuyển quyền điều khiển sang LiDAR |
| `GOPR0572_frame_000255_blur2.jpg` (Blur L2) | **0.793** | **Degraded** | **0.793** | Cảnh báo camera nhòe; nhắc kiểm tra ống kính |
| `GOPR0572_frame_000255_noise3.jpg` (Noise L3) | **0.108** | **Critical** | **0.108** | Lỗi cảm biến ISO/nhiệt; kích hoạt chế độ an toàn |
| `GOPR0572_frame_000255_dark2.jpg` (Dark L2) | **0.698** | **Degraded** | **0.698** | Giảm tốc độ an toàn; bật đèn chiếu xa bổ sung |
| `GOPR0572_frame_000255_overexposure2.jpg` | **0.843** | **Healthy** | **0.843** | Lọc ánh sáng chói; xe và người vẫn an toàn |

---

## 7. Bằng chứng chạy & Tái lập kết quả

Toàn bộ các module do tôi phụ trách đều đã được tích hợp hoàn chỉnh và có thể tái lập thông qua các lệnh terminal chuẩn trong repo:

```bash
# 1. Kích hoạt môi trường và kiểm tra cấu trúc thư mục
python -m data.download.download
python -m data.preprocessing.preprocess
python -m data.preprocessing.split

# 2. Kiểm thử Sensor Degradation Engine sinh 14 biến thể suy biến
python -m data.corruption.corrupt

# 3. Chạy pipeline tạo metadata chuẩn hoá theo Data Contract
python -m data.build_metadata
# -> Tạo file dataset/metadata.csv với đủ 16 000 dòng

# 4. Tái lập trên Kaggle GPU (sinh features, metrics và các bộ examples)
python kaggle/build.py
kaggle kernels push -p kaggle/kernel --accelerator NvidiaTeslaT4
kaggle kernels output nguyendangthuc11/camera-health-acdc-rain -p outputs/kaggle

# 5. Sinh hình ảnh trực quan hoá và phân tích ground truth diff
python -m reports.make_figures

# 6. Kiểm tra dự đoán và điều phối trọng số camera
python -m health_model.predict outputs/results/demo_predict.csv
```

### Thống kê kiểm chứng thực thi từ hệ thống:

```text
Reading features from: D:\K4-Track4-Day04-LiDAR-Sensor-Reality-Sprint\outputs\features\features.csv ...
Successfully generated 16000 metadata rows -> D:\K4-Track4-Day04-LiDAR-Sensor-Reality-Sprint\dataset\metadata.csv

=== DATASET PIPELINE VERIFICATION SUMMARY ===
Total Images Processed   : 16 000
- Clean Reference Images : 1 000 (Train: 400, Val: 100, Test: 500)
- Synthetic Degradations : 14 000 (14 variants per clean image)
- Real Rain ACDC Frames  : 1 000 (Unpaired OOD evaluation)
- Verified Labels (n>=3) : 11 280 labels
Data Contract Status     : PASSED (Schema matches README.md specifications)
Deterministic Seed Check : PASSED (CRC32 checksum verified across platforms)
```

---

## 8. Kết luận & Hướng phát triển cá nhân

Phân hệ **Dataset & Data Pipeline** đã hoàn thành xuất sắc các mục tiêu đề ra:
1. **Thiết lập nền tảng dữ liệu vững chắc:** Xử lý và chuẩn hóa 16 000 ảnh từ tập dữ liệu lái xe thực tế ACDC, loại bỏ hoàn toàn hiện tượng rò rỉ dữ liệu qua chiến lược phân chia theo chuỗi hành trình (sequence-based split).
2. **Tiên phong phương pháp gán nhãn ADAS-Centric:** Chứng minh bằng thực nghiệm rằng các độ đo pixel truyền thống (PSNR/SSIM) không tương thích với an toàn xe tự hành; đồng thời xây dựng thành công bộ suy biến Pixel-Aligned cho phép tính toán Ground Truth Health từ hiệu năng của mạng nơ-ron nhận thức.
3. **Đóng góp cho hệ thống chung:** Cung cấp Data Contract chuẩn xác làm cầu nối cho cả 5 thành viên trong nhóm, hỗ trợ đắc lực việc huấn luyện mô hình XGBoost đạt MAE 0.146 và triển khai cơ chế điều phối trọng số cảm biến an toàn.

**Hướng mở rộng tiếp theo:**
- Nâng cấp bộ suy biến cục bộ (Local Degradation): phát triển thêm module sinh vết bẩn bùn đất văng lên kính chắn gió, mô phỏng dải gạt nước động và phản quang mặt đường ướt để xóa nhòa khoảng cách miền (Domain Gap) giữa dữ liệu tổng hợp và điều kiện thời tiết thực tế ngoài đời sống.
