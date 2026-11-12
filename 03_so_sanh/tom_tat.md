# So sánh mô hình — đơn giá đất ở TP.HCM (đ/m²)

Cùng dữ liệu, đặc trưng và cách chia của A1 (SEED 42, val 20%, target log1p). Siêu tham số chọn bằng 6 cấu hình ngẫu nhiên, **chỉ dựa trên Val MAPE**. LightGBM chạy lại **khớp manifest**: best_iter 3999, Test MAPE 9,51%, R² 0,9492.

Chỉ số chính: **MAPE** (thấp là tốt), **APE<10%** và **APE<15%** (tỷ lệ dòng, cao là tốt). Các chỉ số khác (R², MAE, RMSE, MdAPE, sai số có dấu, APE<20%) có trong `ket_qua.xlsx`.

**Tập test (tstd, n = 693): giá trị [KTC 95%] và hiệu so với LightGBM [KTC 95%], bootstrap theo cặp 5000 lần**

| Mô hình | MAPE % | APE<10% | APE<15% | Δ MAPE | Δ APE<10% | Δ APE<15% |
|---|---|---|---|---|---|---|
| LightGBM (A1) | **9,51** [8,81–10,21] | **64,1** [60,6–67,5] | **79,1** [76,0–82,1] | — | — | — |
| HistGradientBoosting | 9,77 [9,06–10,50] | 64,9 [61,3–68,4] | 80,1 [77,2–83,1] | +0,25 [−0,17; +0,67] | +0,9 [−2,2; +3,9] | +1,0 [−1,6; +3,6] |
| CatBoost | 11,93 [11,12–12,79] | 57,0 [53,4–60,8] | 71,1 [67,8–74,5] | +2,42 [+1,77; +3,09] | −7,1 [−11,1; −3,2] | −7,9 [−11,3; −4,8] |
| Hedonic OLS | 15,79 [14,79–16,83] | 43,3 [39,7–46,9] | 60,0 [56,4–63,5] | +6,28 [+5,29; +7,30] | −20,8 [−25,3; −16,5] | −19,1 [−23,0; −15,0] |
| Random Forest | 16,52 [15,31–17,77] | 44,4 [40,7–48,1] | 60,3 [56,7–63,9] | +7,01 [+5,88; +8,23] | −19,6 [−24,1; −15,2] | −18,8 [−22,9; −14,6] |

→ LightGBM và HGB **không phân biệt được** trên cả ba chỉ số. CatBoost, Hedonic và RF **kém hơn có ý nghĩa** trên cả ba.

**Test theo nhóm: MAPE / APE<10% / APE<15%**

| Nhóm | n | LightGBM | HGB |
|---|---|---|---|
| HCM1 | 435 | 8,67 / 67,8 / 82,5 | 8,89 / 67,8 / 82,3 |
| HCM2 | 258 | 10,94 / 57,8 / 73,3 | 11,25 / 60,1 / 76,4 |
| Mã có trong train | 62 | 6,26 / 82,3 / 88,7 | 5,91 / 79,0 / 91,9 |
| Mã mới | 631 | 9,83 / 62,3 / 78,1 | 10,15 / 63,5 / 78,9 |

**Ngoài thời gian thật** (huấn luyện trên 40.545 dòng khác q2-2026, kiểm tra trên toàn bộ tstd)

| Mô hình | Gốc: MAPE / APE<10 / APE<15 | Ngoài thời gian: MAPE / APE<10 / APE<15 | Sai số có dấu (ngoài thời gian) |
|---|---|---|---|
| LightGBM | 9,51 / 64,1 / 79,1 | **11,97 / 55,1 / 72,6** | −5,15% (dự đoán thấp) |
| HistGradientBoosting | 9,77 / 64,9 / 80,1 | 11,53 / 57,4 / 73,9 | −3,73% |

**Bỏ nhóm biến (LightGBM, Test): MAPE / APE<10 / APE<15, Δ so với đầy đủ [KTC 95%]**

| Cấu hình | Giá trị | Δ MAPE | Δ APE<10% | Δ APE<15% |
|---|---|---|---|---|
| Đầy đủ (646 đặc trưng) | 9,51 / 64,1 / 79,1 | — | — | — |
| a) bỏ Chi tiết + Yếu tố khác (13) | 10,01 / 61,9 / 77,3 | +0,50 [−0,09; +1,11] | −2,2 [−5,5; +1,3] | −1,7 [−4,5; +1,0] |
| b) bỏ Đường phố (645) | 10,30 / 63,2 / 76,3 | +0,79 [+0,34; +1,24] | −0,9 [−3,9; +2,3] | −2,7 [−5,3; −0,3] |
| c) bỏ cả ba (12) | 11,04 / 59,7 / 75,8 | +1,52 [+0,77; +2,30] | −4,3 [−8,1; −0,4] | −3,3 [−6,4; −0,3] |

→ Riêng Chi tiết + Yếu tố khác (633 cột) **không cải thiện có ý nghĩa** trên chỉ số nào.

**SHAP (tỷ trọng mean|SHAP| trên test)**: Vùng 41,3% · Phường/xã 21,0% · Độ rộng ngõ 10,9% · Diện tích 4,6% · Đường phố 4,1% · Chi tiết 3,5% · TF-IDF 1,1%.

**LightGBM theo phân vị đơn giá (MAPE / APE<10 / APE<15)**: Q1 10,7 / 60,9 / 73,6 · Q2 9,2 / 64,2 / 80,9 · Q3 8,7 / 68,2 / 83,2 · Q4 9,5 / 63,0 / 78,6. Quận yếu nhất (n ≥ 10, MAPE / APE<10): Thủ Đức 15,2 / 57,1 · Tân Uyên 14,5 / 28,6 · Vũng Tàu 14,4 / 46,2 · Bình Chánh 14,2 / 41,2.

## Cảnh báo / bất thường
1. **Kết quả gốc lạc quan.** Có 2.921 dòng q2-2026 trong train, và 62 mã của test đã có trong train (APE<10% nhóm này 82%). Kiểm tra ngoài thời gian làm **APE<10% giảm 9 điểm** (64 → 55), MAPE tăng lên ~12%, mô hình dự đoán thấp có hệ thống ~5%. Test có Quý_Năm q2-2026, là mức chưa gặp khi huấn luyện.
2. Không có cơ sở thống kê để nói LightGBM tốt hơn HGB.
3. **CatBoost và HGB có thể bị đánh giá thấp.** Mọi cấu hình CatBoost và 3/6 cấu hình HGB đều chạm trần 5000 vòng (chưa hội tụ).
4. **RF và HGB yếu ở biến phân loại nhiều mức.** RF dùng mã hoá thứ tự (sklearn RF không có categorical native) cho Đường phố (2701 mức) và Phường (173). HGB native chỉ nhận ≤ 255 mức, nên giữ 254 mức phổ biến nhất. Không dùng GradientBoostingRegressor vì nó không có categorical native (phải one-hot ~2900 cột) và chạy đơn luồng, quá chậm.
5. **Siêu tham số chọn theo Val MAPE**, chưa tối ưu riêng cho APE<10/15%.
6. **SHAP**: dùng `pred_contrib` của LightGBM (TreeSHAP) vì chưa cài gói `shap`. Tổng SHAP lệch tối đa 1,8e-3 log (≈0,18%). Đã thêm nhóm Hình dáng. Vùng và Phường gần trùng thông tin nên chia nhau đóng góp.
7. **Hedonic**: biến quy mô ở dạng log, không có cột cộng tuyến hoàn hảo, R²(log, train) 0,922. Có p-value thường và p-value vững HC1.
8. Có 13/32 quận trong test với n < 10 (đã gắn nhãn), không đủ để kết luận riêng.

Tệp: `03_so_sanh/ket_qua.xlsx` (sheet 3_* tham số/hệ số, 4–10), `03_so_sanh/bieu_do/bd1–bd4_*.png`, `03_so_sanh/predictions/predictions_*.csv`, `03_so_sanh/phan_tich.py`; mã huấn luyện ở `02_mo_hinh/a1_common.py`, `02_mo_hinh/chay_mo_hinh.py`, nhật ký `02_mo_hinh/log_chay_mo_hinh.txt`.
