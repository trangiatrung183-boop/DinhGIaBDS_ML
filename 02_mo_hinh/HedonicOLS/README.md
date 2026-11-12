# Hedonic OLS (baseline)

Hồi quy tuyến tính kinh điển trong định giá bất động sản, làm mốc so sánh cho các mô hình học máy.

| | Val (n = 8.694) | Test (n = 693) |
|---|---|---|
| MAPE % | 17,76 | **15,79** [14,79–16,83] |
| APE<10% | 40,1 | 43,3 |
| APE<15% | 56,0 | 60,0 |
| R² | 0,821 | 0,862 |

**Đặc tả:**

```
log1p(đơn giá) ~ ln(Diện tích) + ln(Mặt tiền) + ln(Chiều dài) + Số mặt tiền
               + ln1p(Độ rộng ngõ) + ln1p(Khoảng cách đến đường chính)
               + one-hot(Xã/Phường, Vùng, Vị trí trong khung giá, Lợi thế kinh doanh, hinh_dang_sach, Quý_Năm)
```

n = 34.772, k = 234 tham số, R² (thang log, train) = 0,922. Không có cột cộng tuyến hoàn hảo. Không dùng Đường phố, Chi tiết, Yếu tố khác.

| Tệp | Nội dung |
|---|---|
| `ket_qua/he_so.csv` | hệ số, SE, t, p-value thường và p-value vững HC1, % thay đổi đơn giá (exp(b) − 1) |
| `ket_qua/ghi_chu_dac_ta.csv` | đặc tả, mức tham chiếu của từng biến phân loại |
| `ket_qua/tham_so.json` | đặc tả, n, k, R² |
| `ket_qua/chi_so_danh_gia.csv` | chỉ số theo tập và vùng |
| `ket_qua/predictions_val.csv`, `ket_qua/predictions_test.csv` | giá thực tế, dự đoán, APE % từng dòng |

Code: `02_mo_hinh/chay_mo_hinh.py` (mục HEDONIC OLS). Checkpoint gốc: `02_mo_hinh/checkpoints/m_hedonic.joblib`.
