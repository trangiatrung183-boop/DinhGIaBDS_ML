# TabPFN — chạy thử trên dữ liệu đơn giá đất TP.HCM (đ/m²)

Ngày chạy: 12/11/2026. Dùng cùng dữ liệu sạch (`output/*_clean.xlsx`), cùng cách chia của A1 (SEED 42, val 20%, target `log1p`) và cùng phép bootstrap theo cặp như `so_sanh/tom_tat.md`. LightGBM tính lại khớp bảng cũ: 9,51 [8,81–10,21].

**Cấu hình TabPFN.** Dùng **TabPFN v2** (gói `tabpfn` 9.2.0, GPU RTX 3050 4 GB), không chọn siêu tham số theo test.
- **Ngữ cảnh:** TabPFN không huấn luyện mà dùng dữ liệu train làm "ngữ cảnh", tối đa khoảng 10.000 dòng. Vì vậy lấy **3 tập con ngẫu nhiên, mỗi tập 10.000 dòng** từ train 34.772 dòng, rồi lấy trung bình dự đoán trên thang log.
- **Đặc trưng: 48 cột.** Gồm 6 cột số gốc, 33 cột Chi tiết của A1, Quý_Năm đổi thành số, target encoding và số lần xuất hiện cho Phường/xã và Đường phố, mã thứ tự cho Vùng, VT, Lợi thế KD và Hình dáng. Không dùng 600 cột TF-IDF.
- **Val:** đánh giá trên mẫu cố định 2.000/8.694 dòng để giới hạn thời gian. LightGBM và HGB được tính trên đúng mẫu đó. Test dùng đủ **693 dòng**.
- **Giấy phép:** v2.5 trở lên cần đăng nhập Prior Labs để chấp nhận giấy phép nên chưa chạy.

## Kết quả chính — tập test tstd (n = 693), KTC 95% bootstrap theo cặp 5000 lần

| Mô hình | MAPE % | APE<10% | APE<15% | Δ MAPE vs LGB | Δ APE<10% vs LGB | Δ APE<15% vs LGB |
|---|---|---|---|---|---|---|
| LightGBM (A1) | **9,51** [8,81–10,21] | 64,1 [60,6–67,5] | 79,1 [76,1–82,1] | — | — | — |
| HistGradientBoosting | 9,77 [9,06–10,50] | **64,9** [61,3–68,4] | **80,1** [77,2–83,1] | +0,25 [−0,17; +0,67] | +0,9 [−2,2; +3,9] | +1,0 [−1,6; +3,6] |
| **TabPFN v2 (3×10.000 dòng)** | 12,33 [11,44–13,28] | 56,1 [52,4–59,9] | 70,7 [67,2–74,2] | **+2,81 [+2,08; +3,58]** | **−7,9 [−12,0; −3,8]** | **−8,4 [−11,8; −5,1]** |
| TabPFN v2 (1×10.000 dòng) | 12,63 | 54,3 | 69,4 | +3,12 [+2,36; +3,93] | −9,8 | −9,7 |
| TabPFN v2 (1×2.000 dòng, thử) | 13,70 | 50,2 | 66,8 | +4,19 [+3,35; +5,08] | −13,9 | −12,3 |

→ **TabPFN kém hơn LightGBM và HGB một cách có ý nghĩa trên cả ba chỉ số chính.** Khoảng tin cậy của các hiệu số đều không chứa 0. TabPFN đứng giữa: tốt hơn Hedonic OLS và Random Forest (MAPE 15,8 và 16,5), xấp xỉ CatBoost (11,93).

Val (mẫu 2.000 dòng): LightGBM 8,38 · HGB 8,70 · TabPFN 12,70. Xếp hạng giống trên test.

**Thêm dữ liệu ngữ cảnh có giúp ích, nhưng ít dần:**

| Ngữ cảnh | MAPE test |
|---|---|
| 1 × 2.000 dòng | 13,70 |
| 1 × 10.000 dòng | 12,63 |
| 3 × 10.000 dòng | 12,33 |

Từ 2 lên 3 tập con, MAPE val và test gần như không đổi (xem sheet `6_so_tap_con`).

## Theo nhóm (test): MAPE / APE<10%

| Nhóm | n | LightGBM | TabPFN |
|---|---|---|---|
| HCM1 | 435 | 8,67 / 67,8 | 11,71 / 57,9 |
| HCM2 | 258 | 10,94 / 57,8 | 13,36 / 53,1 |
| Mã có trong train | 62 | 6,26 / 82,3 | 12,89 / 64,5 |
| Mã mới | 631 | 9,83 / 62,3 | 12,27 / 55,3 |
| Q1 đơn giá thấp nhất | 174 | 10,65 / 60,9 | 12,46 / 57,5 |
| Q4 đơn giá cao nhất | 173 | 9,54 / 63,0 | **14,24 / 44,5** |

Ở các quận yếu nhất (n ≥ 10), TabPFN có MAPE cao hơn LightGBM nhiều: Thủ Đức 24,4 · Bình Chánh 23,3 · Tân Uyên 18,3 · Vũng Tàu 18,2.

## Ngoài thời gian thật (bỏ q2-2026 khỏi train, kiểm tra trên toàn bộ tstd)

| Mô hình | Gốc: MAPE / APE<10 / APE<15 | Ngoài thời gian: MAPE / APE<10 / APE<15 | Sai số có dấu (ngoài thời gian) |
|---|---|---|---|
| LightGBM | 9,51 / 64,1 / 79,1 | 11,97 / 55,1 / 72,6 | −5,15% |
| HistGradientBoosting | 9,77 / 64,9 / 80,1 | **11,53 / 57,4 / 73,9** | −3,73% |
| TabPFN v2 | 12,33 / 56,1 / 70,7 | 13,89 / 48,3 / 65,4 | **+1,91%** |

## Tài nguyên (đo thật trên máy: i5-12500H, RAM 16 GB, RTX 3050 4 GB)

| Lượt | Thời gian | VRAM đỉnh | RAM tiến trình |
|---|---|---|---|
| Thử: ngữ cảnh 2.000, dự đoán 9.387 dòng | 5,6 phút | 2,4 GB | 2,1 GB |
| Chính: 3 × 10.000, mỗi lượt dự đoán 2.693 dòng | 33 phút (khoảng 11 phút mỗi tập con) | 3,2 GB | 1,9 GB |
| Ngoài thời gian: 3 × 10.000, dự đoán 693 dòng | 10 phút | 3,2 GB | 1,0 GB |

Để so sánh: LightGBM mất 144 giây cho cả huấn luyện lẫn dự đoán.

**Giới hạn của máy:**
- Lô dự đoán 1.024 dòng với ngữ cảnh 10.000 cần 5,7 GB VRAM. Mức này vượt 4 GB nên tràn sang RAM và chậm gấp đôi.
- Chế độ `fit_with_cache` làm hết RAM 16 GB.
- Dự đoán toàn bộ val 8.694 dòng với ngữ cảnh 10.000 sẽ mất khoảng 36 phút cho mỗi tập con. Đây là lý do val chỉ dùng mẫu 2.000 dòng.

## Nhận xét

1. **TabPFN không thay được LightGBM hay HGB trên bài này.** Nó kém khoảng 2,8 điểm MAPE và khoảng 8 điểm APE<10%, trong khi chậm hơn hàng chục lần. Bài này có khoảng 35.000 dòng và biến vị trí có rất nhiều mức (2.701 đường, 173 phường). Đây là kiểu dữ liệu mà cây tăng cường mạnh, còn TabPFN v2 phải nén thông tin vị trí thành target encoding.
2. **TabPFN chỉ thấy khoảng 29% train mỗi lượt** (10.000/34.772 dòng). Điều này giải thích vì sao nó **không được lợi ở nhóm "mã có trong train"** (12,89%, so với 6,26% của LightGBM). Nhóm đó hưởng lợi do LightGBM ghi nhớ mã tài sản cũ. Nhìn theo hướng khác, kết quả của TabPFN ít bị lạc quan giả hơn.
3. **TabPFN yếu nhất ở phân khúc giá cao (Q4)**, với APE<10% chỉ 44,5%. Nó cũng yếu ở các quận có giá phân tán như Thủ Đức và Bình Chánh. Mô hình có xu hướng kéo dự đoán về mức trung bình.
4. **Điểm đáng chú ý ở bài ngoài thời gian:**
   - TabPFN gần như **không lệch có hệ thống**: sai số có dấu +1,9%, so với −5,2% của LightGBM.
   - Mức suy giảm khi chuyển sang ngoài thời gian nhỏ hơn: MAPE tăng 1,6 điểm, so với 2,5 điểm của LightGBM.
   - Dù vậy, MAPE tuyệt đối vẫn kém nhất (13,89).
   - Ý tưởng có thể thử: dùng TabPFN làm thành phần phụ trong ensemble hoặc để hiệu chỉnh độ lệch theo thời gian, thay vì làm mô hình chính.
5. **Hạn chế của lần thử này:**
   - Chưa chạy TabPFN v2.5 trở lên (hỗ trợ tới 50.000 dòng), vì cần chấp nhận giấy phép qua tài khoản Prior Labs, và giấy phép này phi thương mại.
   - Chưa thử fine-tune.
   - Val chỉ là mẫu 2.000 dòng.
   - Cách mã hoá Phường/Đường phố là một lựa chọn chưa được tối ưu.

## Tệp

| Tệp | Nội dung |
|---|---|
| `ket_qua_tabpfn.xlsx` | Các sheet: 1 cấu hình · 2 tài nguyên · 3 đánh giá Val/Test theo vùng · 4 mã tài sản · 5 bootstrap so với LGB và HGB · 6 số tập con · 7 ngoài thời gian · 8 phân vị và quận |
| `predictions_test.csv`, `predictions_val.csv`, `predictions_ngoai_thoi_gian.csv` | Dự đoán của LGB, HGB và TabPFN theo từng dòng |
| `bd_tabpfn_so_sanh.png` | Biểu đồ 3 chỉ số chính kèm KTC 95% |
| `chay_tabpfn.py` | Chạy mô hình: `pilot` để chạy thử, `main` để chạy chính (có checkpoint) |
| `bao_cao_tabpfn.py` | Xuất báo cáo từ checkpoint |
| `log.txt` | Nhật ký thời gian, VRAM và RAM |
