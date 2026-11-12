# CatBoost

| | Val (n = 8.694) | Test (n = 693) |
|---|---|---|
| MAPE % | 11,87 | **11,93** [11,12–12,79] |
| APE<10% | 56,8 | 57,0 |
| APE<15% | 72,4 | 71,1 |
| R² | 0,897 | 0,937 |

Kém hơn LightGBM có ý nghĩa trên cả ba chỉ số (Δ MAPE test +2,42 [+1,77; +3,09]).

**Cấu hình chọn** (cấu hình 1/6): `depth=8`, `learning_rate=0.1`, `l2_leaf_reg=3`, `one_hot_max_size=10`; `iterations=5000`, dừng sớm 200 vòng trên Val → 4.997 vòng, 478 giây.

**Biến phân loại:** categorical native của CatBoost (ordered target statistics) cho cả 7 cột phân loại.

**Lưu ý:** mọi cấu hình đều chạm (hoặc gần chạm) trần 5.000 vòng, chưa hội tụ. Kết quả này có thể đánh giá thấp CatBoost.

## Tệp

| Tệp | Nội dung |
|---|---|
| `ket_qua/tham_so.json` | siêu tham số được chọn, số vòng, thời gian huấn luyện |
| `ket_qua/tim_kiem_sieu_tham_so.csv` | 6 cấu hình ngẫu nhiên đã thử và Val MAPE của từng cấu hình |
| `ket_qua/chi_so_danh_gia.csv` | chỉ số theo tập (Train/Val/Test) và vùng (HCM1/HCM2) |
| `ket_qua/predictions_val.csv`, `ket_qua/predictions_test.csv` | giá thực tế, dự đoán, APE % từng dòng |

Code: `02_mo_hinh/chay_mo_hinh.py` (mục CATBOOST). Checkpoint gốc: `02_mo_hinh/checkpoints/m_catboost.joblib`.
