# Báo cáo cá nhân — Phan Trọng Hoàn (2A202602954)

**Vai trò:** Image Quality / Feature Engineering · **Repo nhóm:** https://github.com/ThucNguyen1705/K4-Track4-Day04-LiDAR-Sensor-Reality-Sprint · **Báo cáo nhóm:** [../REPORT.md](../REPORT.md)

## 1. Phần việc của tôi
Phụ trách Image Quality / Feature Engineering: triển khai trích xuất 11 đặc trưng IQA toàn ảnh và 36 đặc trưng theo lưới 3×3 trong [features/](../../features/). Pipeline resize ảnh về 1280×720, chuẩn hóa các đặc trưng độ sáng và xuất dữ liệu cho các bước huấn luyện/đánh giá. Tôi cũng thực hiện kiểm chéo feature trên ảnh mưa thật với bộ CSV độc lập.

Các đặc trưng bao quát độ nét, độ sáng, tương phản, nhiễu, cạnh, entropy và màu sắc để biểu diễn nhiều dạng suy giảm khác nhau. Phần grid chia ảnh thành 9 vùng, giúp giữ thêm thông tin về vị trí của biến đổi thay vì chỉ gộp thành giá trị trung bình toàn ảnh. Kết quả được ghép với metadata theo `image_id` để các bước sau sử dụng nhất quán.

## 2. Nguồn tham khảo (paper / repo)
- J. Immerkær, *Fast Noise Variance Estimation*, CVIU 1996 — ước lượng nhiễu.
- D. Hasler và S. Süsstrunk, *Measuring Colourfulness in Natural Images*, SPIE 2003 — thống kê màu/colorfulness.
- ACDC: C. Sakaridis, D. Dai, L. Van Gool, *The Adverse Conditions Dataset with Correspondences*, ICCV 2021 — dữ liệu ảnh điều kiện bất lợi.

Phương pháp Immerkær được dùng làm cơ sở cho `noise_estimate`; công thức colorfulness của Hasler và Süsstrunk hỗ trợ đặc trưng màu. ACDC cung cấp ảnh tham chiếu và ảnh trong điều kiện mưa để xây dựng, kiểm tra pipeline trên các cảnh giao thông thực tế. Các phép đo được triển khai trong package `features/` của repo.

## 3. Cách nhóm đo metric
Đánh giá feature bằng độ nhạy theo degradation ([feature_response.png](../figures/feature_response.png)), tương quan Spearman với health ([feature_health_corr.csv](../figures/feature_health_corr.csv)) và kiểm chéo trên 1.000 ảnh mưa thật ([feature_crosscheck.csv](../../outputs/results/feature_crosscheck.csv)). Trong kiểm chéo, 10/11 feature có Spearman xấp xỉ từ 0.94 trở lên; `noise_estimate` đạt 0.818. Health của mô hình nhóm được gán bằng recall detector trên ảnh degraded chia recall trên ảnh sạch cùng cảnh. Benchmark chung của nhóm: XGBoost đạt MAE 0.146, R² 0.621 và Spearman 0.746 trên test ([results.csv](../../outputs/results/results.csv)); đây là kết quả mô hình chung, không phải metric riêng của feature extractor.

Biểu đồ feature response biểu diễn mức thay đổi của từng feature theo loại và mức degradation so với ảnh sạch; qua đó có thể quan sát những feature nhạy với blur, noise, dark hoặc overexposure. Spearman trong kiểm chéo đo mức độ đồng biến theo thứ hạng giữa hai cách trích xuất trên cùng ảnh, không yêu cầu giá trị tuyệt đối phải giống nhau. Cần lưu ý các chỉ số MAE, R² và Spearman của XGBoost phản ánh toàn bộ pipeline dự đoán health, không đánh giá riêng từng thuật toán feature.

## 4. Failure case
Feature toàn ảnh chưa mô tả tốt suy giảm cục bộ như cần gạt nước hoặc giọt nước trên kính; do đó mô hình nhóm thường đánh giá ảnh mưa thật lành mạnh. Ngoài ra, `noise_estimate` nhạy với độ phân giải (Spearman 0.818 trong kiểm chéo), còn Laplacian variance có median lệch khoảng 2.28 lần giữa ảnh 1920×1080 và pipeline 1280×720.

Đây là hạn chế của việc mô tả cảnh bằng thống kê tổng hợp: vùng ảnh bị che khuất nhỏ có thể không làm thay đổi đáng kể giá trị trung bình toàn khung hình. Kiểm chéo cho thấy độ phân giải ảnh ảnh hưởng đến một số phép đo, vì vậy cùng một cảnh có thể cho độ lớn feature khác nhau nếu bỏ qua bước resize. Proxy recall trên ảnh mưa thật cũng chỉ là chỉ dấu tham khảo, không phải ground truth độc lập.

## 5. Engineering decision & trade-off
Resize về 1280×720 và thống nhất đơn vị feature để các nguồn ảnh so sánh được. Bổ sung grid 3×3 để giữ thông tin suy giảm theo vùng; tuy nhiên, benchmark nhóm cho thấy XGBoost với 47 feature (11 global + 36 grid) có MAE 0.155, kém hơn bản 11 feature (MAE 0.146). Vì vậy grid hiện hữu ích cho mở rộng/khảo sát lỗi cục bộ nhưng chưa được chọn cho mô hình chính.

Tôi giữ nhóm feature toàn ảnh gọn để mô hình chính có đầu vào ổn định và dễ so sánh giữa các nguồn dữ liệu. Grid là hướng bổ sung nhằm tăng khả năng biểu diễn lỗi cục bộ, nhưng số chiều tăng từ 11 lên 47 có thể làm mô hình nhạy hơn với tập huấn luyện hiện tại. Có thể đánh giá lại grid khi bổ sung dữ liệu hoặc các dạng degradation cục bộ được mô phỏng phù hợp hơn.

## 6. Bằng chứng chạy
```bash
python -m features.extract_features --workers 4
python -m features.crosscheck
```
Kết quả và hình tham chiếu: [feature_crosscheck.log](../../outputs/logs/feature_crosscheck.log), [feature_crosscheck.png](../figures/feature_crosscheck.png), [feature_response.png](../figures/feature_response.png).

Lệnh trích xuất đọc metadata theo cấu hình dự án, tính feature cho các ảnh hợp lệ và ghi CSV vào `outputs/features/features.csv`. Có thể dùng `--no-grid` để xuất riêng các feature toàn ảnh hoặc thay số worker bằng `--workers`. Lệnh kiểm chéo so sánh dữ liệu với `rain_features_csv/` và lưu bảng, dự đoán cùng biểu đồ trong `outputs/results/`.
