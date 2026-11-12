# HistGradientBoosting (scikit-learn)

| | Val (n = 8.694) | Test (n = 693) |
|---|---|---|
| MAPE % | 8,95 | **9,77** [9,06–10,50] |
| APE<10% | 69,7 | 64,9 |
| APE<15% | 82,8 | 80,1 |
| R² | 0,930 | 0,949 |

Không phân biệt được với LightGBM trên cả ba chỉ số chính (Δ MAPE test +0,25 [−0,17; +0,67]).

**Cấu hình chọn** (cấu hình 4/6): `learning_rate=0.05`, `max_leaf_nodes=255`, `min_samples_leaf=10`, `l2_regularization=0`, `max_features=0.8`; dừng sớm trên Val (bước 100 vòng, kiên nhẫn 300 vòng, tối đa 5.000 vòng) → 3.736 vòng, 1.158 giây.

**Biến phân loại:** dùng categorical native của HGB, nhưng HGB chỉ nhận ≤ 255 mức, nên mỗi cột giữ 254 mức phổ biến nhất và gộp phần còn lại (ảnh hưởng Đường phố, 2.701 mức).

**Lưu ý:** 3/6 cấu hình chạm trần 5.000 vòng, nên HGB có thể còn cải thiện được.

## Tệp

| Tệp | Nội dung |
|---|---|
| `ket_qua/tham_so.json` | siêu tham số được chọn, số vòng, thời gian huấn luyện |
| `ket_qua/tim_kiem_sieu_tham_so.csv` | 6 cấu hình ngẫu nhiên đã thử và Val MAPE của từng cấu hình |
| `ket_qua/chi_so_danh_gia.csv` | chỉ số theo tập (Train/Val/Test) và vùng (HCM1/HCM2) |
| `ket_qua/predictions_val.csv`, `ket_qua/predictions_test.csv` | giá thực tế, dự đoán, APE % từng dòng |

Code: `02_mo_hinh/chay_mo_hinh.py` (mục HIST GRADIENT BOOSTING). Checkpoint gốc: `02_mo_hinh/checkpoints/m_hgb.joblib`.
