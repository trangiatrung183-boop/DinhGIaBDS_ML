# LightGBM — phiên bản A1 (mô hình chính)

| | Val (n = 8.694) | Test (n = 693) |
|---|---|---|
| MAPE % | 8,69 | **9,51** [8,81–10,21] |
| APE<10% | 71,2 | 64,1 |
| APE<15% | 83,7 | 79,1 |
| R² | 0,930 | 0,949 |

**A1** dựng trên pipeline gốc của công ty (phiên bản A), với các thay đổi: dùng dữ liệu đã làm sạch (`data/processed/`), chỉ dùng LightGBM, bỏ mọi biến tính từ đơn giá (`Don_gia_TB`, `Min`, `Max`, `DG_*`…) và ba cột `Chi_so_*`, thay `Hình dáng` bằng `hinh_dang_sach`. Bảng so sánh đầy đủ ở ô đầu notebook.

**Tham số:** `objective=regression`, `learning_rate=0.03`, `num_leaves=255`, `min_child_samples=5`, `feature_fraction=0.8`, `bagging_fraction=0.8`, `bagging_freq=1`, `lambda_l2=1.0`, `n_estimators=12000`, dừng sớm 200 vòng trên Val → 3.999 vòng. Target `log1p`. 646 đặc trưng, 7 biến phân loại dùng categorical native.

## Tệp

| Tệp | Nội dung |
|---|---|
| `pipeline_LightGBM.ipynb` | notebook huấn luyện + đánh giá + lưu model (bản gốc: `A1/pipeline_HCM_A1.ipynb`) |
| `model/lightgbm_model.txt.gz` | booster LightGBM đã huấn luyện (nén; giải nén ra `lightgbm_model.txt`, 110 MB) |
| `model/preprocessor.joblib` | trạng thái tiền xử lý: trung vị, danh mục, TF-IDF, SVD |
| `model/manifest.json` | danh sách đặc trưng, tham số, chỉ số, SHA-1 dữ liệu, phiên bản thư viện |
| `ket_qua/ket_qua_danh_gia_LightGBM.xlsx` | chỉ số đánh giá theo tập và vùng |
| `ket_qua/test_predictions_LightGBM.xlsx`, `ket_qua/val_predictions_LightGBM.xlsx` | dự đoán từng dòng |
| `ket_qua/feature_importance_LightGBM.csv` | độ quan trọng đặc trưng (gain) |
| `ket_qua_so_sanh/` | cùng mô hình được huấn luyện lại trong `chay_mo_hinh.py` để so sánh (khớp manifest: 3.999 vòng, Test MAPE 9,51%); cùng định dạng với các mô hình khác |

Notebook có hàm nạp model đã lưu và dự đoán cho dữ liệu mới (ô cuối), yêu cầu dữ liệu đã làm sạch theo `01_lam_sach/`.

Tên tệp kết quả đã đổi từ `*_HCM_A1.*` thành `*_LightGBM.*`. Bên trong tệp vẫn ghi `HCM_A1` / `LightGBM (A1)` như lúc chạy.
