# Random Forest (scikit-learn)

| | Val (n = 8.694) | Test (n = 693) |
|---|---|---|
| MAPE % | 16,77 | **16,52** [15,31–17,77] |
| APE<10% | 47,9 | 44,4 |
| APE<15% | 61,7 | 60,3 |
| R² | 0,777 | 0,835 |

Kém hơn LightGBM có ý nghĩa trên cả ba chỉ số (Δ MAPE test +7,01 [+5,88; +8,23]).

**Cấu hình chọn** (cấu hình 5/6): `n_estimators=300`, `max_features=0.33`, `min_samples_leaf=1`, `max_depth=None`; 28 giây.

**Biến phân loại:** RandomForest của scikit-learn không có categorical native, nên dùng mã hoá thứ tự (OrdinalEncoder). Cách này bất lợi cho cột nhiều mức như Đường phố (2.701 mức) và Phường (173 mức), là lý do chính khiến RF yếu.

## Tệp

| Tệp | Nội dung |
|---|---|
| `ket_qua/tham_so.json` | siêu tham số được chọn, số vòng, thời gian huấn luyện |
| `ket_qua/tim_kiem_sieu_tham_so.csv` | 6 cấu hình ngẫu nhiên đã thử và Val MAPE của từng cấu hình |
| `ket_qua/chi_so_danh_gia.csv` | chỉ số theo tập (Train/Val/Test) và vùng (HCM1/HCM2) |
| `ket_qua/predictions_val.csv`, `ket_qua/predictions_test.csv` | giá thực tế, dự đoán, APE % từng dòng |

Code: `02_mo_hinh/chay_mo_hinh.py` (mục RANDOM FOREST). Checkpoint gốc: `02_mo_hinh/checkpoints/m_rf.joblib`.
