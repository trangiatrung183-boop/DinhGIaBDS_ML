# TabPFN v2

Mô hình nền tảng cho dữ liệu bảng (Prior Labs). TabPFN không huấn luyện theo cách thông thường mà dùng dữ liệu train làm "ngữ cảnh" (tối đa khoảng 10.000 dòng), nên lấy 3 tập con ngẫu nhiên, mỗi tập 10.000 dòng từ train, rồi lấy trung bình dự đoán trên thang log.

| | Val (mẫu 2.000 dòng) | Test (n = 693) |
|---|---|---|
| MAPE % | 12,70 | **12,33** [11,44–13,28] |
| APE<10% | 53,0 | 56,1 |
| APE<15% | 69,9 | 70,7 |
| R² | 0,908 | 0,912 |

Kém hơn LightGBM có ý nghĩa trên cả ba chỉ số (Δ MAPE test +2,81 [+2,08; +3,58]). Thêm ngữ cảnh có giúp nhưng ít dần: 1 × 2.000 dòng → MAPE 13,70; 1 × 10.000 → 12,63; 3 × 10.000 → 12,33.

**Đặc trưng: 48 cột** (khác các mô hình cây): 6 cột số gốc, 33 cột Chi tiết của A1, Quý_Năm đổi thành số, target encoding (K-fold) và số lần xuất hiện cho Phường/xã và Đường phố, mã thứ tự cho Vùng, VT, Lợi thế KD và Hình dáng. Không dùng 600 cột TF-IDF.

**Môi trường:** gói `tabpfn` 9.2.0 (model TabPFN v2), GPU RTX 3050 4 GB. Các bản v2.5 trở lên cần đăng nhập Prior Labs để chấp nhận giấy phép nên chưa chạy.

## Tệp

| Tệp | Nội dung |
|---|---|
| `tom_tat_tabpfn.md` | tóm tắt đầy đủ, cảnh báo |
| `bang_so_lieu.md` | bảng số liệu kèm KTC 95% |
| `ket_qua_tabpfn.xlsx` | cấu hình, tài nguyên, đánh giá, bootstrap, số tập con, ngoài thời gian, phân vị/quận |
| `predictions_val.csv`, `predictions_test.csv`, `predictions_ngoai_thoi_gian.csv` | dự đoán từng dòng (kèm LightGBM, HGB để đối chiếu) |
| `bd_tabpfn_so_sanh.png` | biểu đồ so sánh |
| `chay_tabpfn.py` | chạy mô hình: `pilot` (đo tài nguyên), `main` (chạy chính + ngoài thời gian) |
| `bao_cao_tabpfn.py` | đọc checkpoint, xuất Excel / CSV / PNG |
| `checkpoints/` | dự đoán đã lưu của từng lần chạy |
| `log.txt`, `run_stdout.txt` | nhật ký |

`chay_tabpfn.py` đọc ma trận đặc trưng `02_mo_hinh/checkpoints/features_main.joblib` và `features_oot.joblib`, nên cần chạy `02_mo_hinh/chay_mo_hinh.py` trước để sinh hai file này.
