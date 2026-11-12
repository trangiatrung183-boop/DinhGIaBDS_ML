# Bước 1 · Làm sạch dữ liệu giá đất HCM_3

Làm sạch tập huấn luyện `train_HCM_3.xlsx` và tập test `tstd_HCM_3.xlsx` thành bộ dữ liệu sẵn sàng cho mô hình. Hai tập dùng chung một bộ rule, nên kết quả nhất quán giữa train và test.

## Chạy nhanh

Từ thư mục gốc repo:

```bash
pip install -r requirements.txt
```

1. Dữ liệu gốc nằm ở `data/raw/`.
2. Mở `01_lam_sach/pipeline_lam_sach_train_HCM_3.ipynb`, chọn **Run All** (1–4 phút, phần lớn là đọc file Excel).
3. Mở `01_lam_sach/pipeline_lam_sach_tstd_HCM_3.ipynb`, chọn **Run All** (thêm 1–2 phút). Notebook test cần bảng tra hệ số hình dáng do bước 2 tạo ra, nên phải chạy train trước.

Kết quả ghi vào `data/processed/`. Notebook tự tìm thư mục gốc repo, nên mở từ đâu cũng chạy được.

> **Chạy trên dữ liệu khác bản gốc:** notebook train so từng số liệu với kỳ vọng đo trên file gốc (`KY_VONG`) và dừng ở chỗ lệch đầu tiên. Với dữ liệu mới, đặt `STRICT = False` trong cell Cấu hình: notebook chỉ in cảnh báo mà không dừng.

## Tệp

```
data/raw/                                    # đầu vào (không sửa)
├── train_HCM_3.xlsx                         # 49.889 dòng × 78 cột
└── tstd_HCM_3.xlsx                          # 710 dòng × 67 cột
01_lam_sach/
├── lam_sach.ipynb                           # toàn bộ rule, dùng chung (chỉ định nghĩa hàm, không tự chạy)
├── pipeline_lam_sach_train_HCM_3.ipynb      # chạy từng bước cho train, có chú thích
└── pipeline_lam_sach_tstd_HCM_3.ipynb       # chạy từng bước cho test
data/processed/                              # sinh ra khi chạy, xoá đi chạy lại được
├── train_HCM_3_clean.xlsx                   # 43.466 dòng × 42 cột
├── tstd_HCM_3_clean.xlsx                    # 693 dòng × 40 cột
├── bao_cao_rule.csv                         # số dòng tác động của từng rule (train)
├── bao_cao_rule_tstd.csv                    # như trên (test)
├── bang_tra_he_so_hinh_dang.csv             # hệ số hình dáng cho từng chuỗi Hình dáng, dựng từ train
├── tstd_dong_bi_loai.csv                    # dòng test bị loại vì trùng
└── tstd_tai_san_co_trong_train.csv          # Mã tài sản của test đã có trong train
```

## Sửa một rule

Mỗi bước B1–B9 là một hàm trong `01_lam_sach/lam_sach.ipynb` (tên `b<số>_...`, ví dụ `b4_hinh_dang`). Sửa hàm đó, rồi chạy lại cả hai notebook pipeline. Nếu rule làm đổi số liệu, cập nhật `KY_VONG` trong notebook train. Mọi can thiệp đều được ghi vào `bao_cao_rule*.csv`.

Phần dưới mô tả chi tiết từng bước trên tập train; khác biệt của tập test ở mục cuối.

## Nguyên tắc

1. Không chắc thì để NaN, không ước lượng, không điền giá trị thay thế.
2. Chỉ sửa khi có bằng chứng rõ ràng; trường hợp còn nghi ngờ thì chỉ gắn cờ.
3. Mọi từ khoá và regex đều đã được đếm trên dữ liệu thật.
4. Mọi can thiệp đều có cờ trong dữ liệu hoặc một dòng trong `bao_cao_rule.csv`.
5. Viết thường cho mọi cột chữ, không chừa cột nào.
6. Cột do công ty suy ra từ cột khác (ba cột chỉ số) được tính lại theo rule của công ty, không nhập tay.

---

## Các bước

| Bước | Nội dung | Tác động |
|---|---|---|
| B0 | Đọc dữ liệu gốc, chọn cột | 26 cột đọc vào |
| B1 | Chuẩn hoá text và viết thường | 14 cột chữ |
| B2 | Khử trùng lặp lần 1 | loại 6.418 dòng |
| B3 | Giữ dòng thiếu Đường phố | giữ 11 dòng |
| B4 | Làm sạch từng cột | số, Quận/Huyện, Đoạn đường, Yếu tố khác, Hình dáng, Chi tiết |
| B5 | Mâu thuẫn VT – KC – Chi tiết | sửa 898 dòng, 2 cờ cảnh báo |
| B6 | Ô kích thước và cột tỷ lệ | sửa 5 dòng, thêm 2 cột tỷ lệ |
| B7 | Khử trùng lặp lần 2 | loại 5 dòng |
| B8 | Bổ sung ba cột chỉ số | `Chi_so_ngo`, `Chi_so_dientich`, `Chi_so_hinhdang` |
| B9 | Kiểm thử và xuất file | 43.466 dòng × 42 cột |

### B0 · Đọc dữ liệu gốc và chọn cột

| Nhóm | Cột |
|---|---|
| 14 cột làm sạch | `Xã/Phường/Thị trấn mới`, `Đường phố`, `Đoạn đường`, `Chi tiết`, `Khoảng cách đến đường chính (m)` (KC), `Vị trí trong khung giá` (VT), `Độ rộng ngõ/ngách nhỏ nhất (Từ đường chính đến BĐS)`, `Diện tích (m2)` (DT), `Kích thước mặt tiền (m)` (MT), `Kích thước chiều dài` (CD), `Số mặt tiền tiếp giáp`, `Hình dáng`, `Lợi thế kinh doanh`, `Yếu tố khác` |
| 9 cột giữ nguyên nội dung | `Ghep`, `Don_gia_TB`, `Min`, `Max`, `Độ lệch chuẩn`, `Số lượng mẫu`, `Tháng`, `Quý_Năm`, `Vùng` |
| 2 cột bổ sung | `Thành phố/Quận/Huyện/Thị xã`, `Mã tài sản` |
| Target | `Đơn giá quyền sử dụng đất (đ/m2)1` |

Các cột phụ (tên bắt đầu bằng `_`) chỉ dùng trong lúc xử lý và bị bỏ trước khi xuất: số dòng trong file gốc (phá hoà), ngày hiệu lực (chọn bản giữ khi trùng), độ rộng, diện tích và chuỗi Hình dáng **gốc chưa làm sạch** (để B8 tính chỉ số), và ba cột chỉ số có sẵn của công ty (chỉ để đối chiếu ở B8).

### B1 · Chuẩn hoá text và viết thường

Áp cho 14 cột chữ (không áp cho cột phụ): chuẩn hoá Unicode NFKC, xoá ký tự vô hình (`U+200B`–`U+200F`, `U+FEFF`, `U+00A0`, `U+2060`), gộp khoảng trắng, viết thường; ô chỉ chứa ký hiệu (`-`, `.`, `*`, `n/a`…) thành NaN.

Lưu ý kỹ thuật: trên pandas 3, cột chữ có dtype `str` chứ không phải `object`. Notebook chọn cột chữ theo điều kiện "không phải số và không phải ngày".

### B2 · Khử trùng lặp lần 1

- **Khoá:** 14 cột làm sạch + target. Không đưa cột thời gian hay cột nhóm vào khoá, vì cùng một tài sản thường được nhập lại ở nhiều tháng.
- **Chọn bản giữ:** bản có `Thời điểm hiệu lực` muộn nhất; nếu hoà thì giữ dòng xuất hiện trước trong file gốc.

Loại **6.418 dòng**, còn 43.471. Có 4.036 nhóm trùng (nhóm lớn nhất 206 bản); 2.598 nhóm khác ngày hiệu lực và 768 nhóm nằm ở hai quý khác nhau.

### B3 · Giữ dòng thiếu `Đường phố`

11 dòng thiếu `Đường phố` được giữ nguyên. Khoá `Ghep` của chúng cũng thiếu tên đường nên không suy ngược được.

### B4 · Làm sạch từng cột

**B4.1 · Cột số và Vị trí trong khung giá**

| Rule | Số dòng |
|---|---|
| `vt5` → `vt4` (VT5 quá ít để tách riêng) | 4 |
| `Số mặt tiền tiếp giáp` = 0 hoặc > 4 → NaN | 21 |
| `Độ rộng ngõ/ngách` = 0 → NaN | 3 |
| `Kích thước mặt tiền` = 0 → NaN | 1 |
| `KC` > 2.500 m → NaN | 1 |

`Độ rộng ngõ/ngách` mang nghĩa kép: với VT1 là bề rộng con đường ở cột `Đường phố`, với VT khác là bề rộng ngõ. Giá trị được giữ nguyên.

**B4.2 · Quận/Huyện:** `thị xã thuận an` → `thành phố thuận an` (cùng một đơn vị có hai tên). 1 dòng; còn 40 giá trị.

**B4.3 · Đoạn đường:** gộp cách viết cùng nghĩa (180 dòng; còn 1.078 giá trị).

| Rule | Ví dụ |
|---|---|
| `đầu đường - cuối đường`, `toàn tuyến` → `trọn đường` | |
| Cụm bắt đầu bằng `bề rộng mặt đường dưới` → `bề rộng mặt đường nhỏ hơn` | `bề rộng mặt đường dưới 9 m` |
| Bỏ khoảng trắng giữa số và `m` | `9 m` → `9m` |
| Bỏ dấu chấm cuối | |

Dạng khoảng `từ 6m đến dưới 9m` giữ nguyên.

**B4.4 · Yếu tố khác:** ô chỉ chứa số → NaN (`0` là ô bỏ trống; `394334534`, `6,000,000,000` là số tiền nhập nhầm cột). 18 dòng.

**B4.5 · Hình dáng → `hinh_dang_sach`** (cột gốc giữ nguyên)

| Phần | Nội dung | Số dòng |
|---|---|---|
| 1/3 | Giá trị là số hoặc một chữ cái → NaN (`156.50`, `7,500,000`, `v`) | 8 |
| 2/3 | Sửa lỗi gõ (`cân dối`, `câm đối`, `cân đôi`, `đối` → `cân đối`; `cân đối5` → `cân đối`; `vạc góc` → `vạt góc`…) | 35 |
| 2/3 | Nhãn không phải hình dáng (`không`, `khá`, `nhà hẻm`) → NaN | 7 |
| 3/3 | Thống nhất cách viết: `chữ l`, `hình l` → `hình chữ l`; các ý trong nhãn ghép cách nhau bằng `; ` (dấu `-`, `,` sau nhãn gốc và dấu ngoặc đều đổi thành `; `) | 218 |

Kết quả: **45 nhãn**, 15 NaN, không còn dấu ngoặc. Tách ý bằng `str.split('; ')`. Không gom nhóm: mức độ (`nở hậu nhẹ`, `tóp hậu ít`) và nhãn có số đo (`nở hậu 10m`) được giữ riêng. Ví dụ: `không cân đối (chữ L)` → `không cân đối; hình chữ l`.

**B4.6 · Chi tiết → tách thành các ý**

Nội dung cột `Chi tiết` được thay bằng chuỗi các ý cách nhau bằng `; `:

| Chữ gốc | Sau khi tách |
|---|---|
| `Nhà hẻm` | `hẻm` |
| `346` | `mt` |
| `57/1A Đường Số 7, KP 5` | `hẻm; đường số 7; khu phố 5` |
| `Khu phố Đông An; Đường D1 - KDC Đông An` | `đường d1; kdc đông an; khu phố đông an` |
| `Nhà mtnb kdc hai thành` | `mt; đường nội bộ; kdc hai thành` |

1. **Thống nhất viết tắt:** `khu dân cư`, `kđc`, `khu dc` → `kdc`; `tđc`, `ktđc` → `khu tái định cư`; `kp` → `khu phố`; `kno` → `khu nhà ở`; `đnb`, `nội bộ` → `đường nội bộ`; `đs 2`, `đường 2` (khi `đường` mở đầu cụm) → `đường số 2`; `tl 37`, `ht 45` → `tl37`, `ht45`.
2. **Ý vị trí đứng đầu:** `mt` khi có chữ `mt`, `2mt`, `mặt tiền`, `mtnb`, `shophouse`, hoặc mảnh đầu là số nhà đơn (`346`) hay dải số nhà (`137-139`); `hẻm` khi có chữ `hẻm`, hoặc số nhà dạng `57/1a`, `102/` khi câu không có chữ mt. Số dạng ngày đứng sau `đường`, `kdc`, `khu phố` (như `đường 3/2`) là tên đường, không phải số nhà hẻm.
3. **Các ý còn lại** xếp theo thứ tự: `đường …` → `kdc …` → `khu …` → `ấp …` → phần còn lại. Tên khu giữ trọn (`kdc ấp 5c`).
4. **Bỏ:** mô tả nhà/đất (`nhà`, `đất`, `nhà nát`, `đất trống`, `biệt thự`, `có thang máy`…), số nhà, đường link, tên đường đứng sau `mt`/`hẻm` khi trùng cột `Đường phố`.

Kết quả: 11.528 → **6.295 giá trị**; 4.406 NaN (4.405 sau B7). Chữ trước khi tách được giữ tạm trong lúc xử lý, vì B5 và B7 cần đọc nó.

### B5 · Mâu thuẫn VT – KC – Chi tiết

Bốn cờ nhận diện đọc trên `Chi tiết` **trước khi tách ý**:

| Cờ | Điều kiện | Số dòng |
|---|---|---|
| `la_hem` | có chữ `hẻm` | 15.674 |
| `co_mat_tien` | `\bmt\b`, `mặt tiền` hoặc `shophouse` | 9.520 |
| `chi_la_mat_tien` | chỉ ghi đúng `mt`, `mặt tiền`, `nhà mt`, `đất mt` | 5.393 |
| `la_so_nha` | chỉ là một số nhà, không có `/` (`346`, `20a`) | 825 |

`co_bang_chung_mat_tien = co_mat_tien | la_so_nha`.

| Rule | Xử lý | Số dòng |
|---|---|---|
| **Sửa 1** | `chi_la_mat_tien` → `VT = vt1`, `KC = 0`, **trừ** khi VT là vt2–vt4 kèm KC cụ thể khác 0 (có thể là mặt tiền của đường ngoài bảng giá nhà nước; chỉ gắn cờ) | đổi VT 150, đổi KC 12, không sửa 61 |
| **Sửa 2** | vt2–vt4 có `KC = 0` nhưng không có `co_bang_chung_mat_tien` → `KC = NaN` | 736 |
| `canh_bao_vt1` | VT1 nhưng có chữ hẻm, hoặc `KC > 0`, hoặc độ rộng < 4 m. Chỉ gắn cờ | 426 |
| `canh_bao_mt_vt234` | có chữ mt nhưng VT là vt2–vt4 và `KC > 0`. Chỉ gắn cờ | 1.805 |

Sửa 1 là rule duy nhất sửa VT **theo bằng chứng** (B4.1 có gộp `vt5` vào `vt4`, nhưng đó là gộp nhãn). Khi soát `canh_bao_vt1`, 22 dòng vừa có chữ hẻm vừa có chữ mt, thường là nhà mặt tiền có hẻm phía sau.

### B6 · Ô kích thước và cột tỷ lệ

- **Sửa 3:** ô kích thước bị nhập đúng bằng diện tích → tính lại từ hai ô còn lại: `MT == DT` (2 dòng) → `MT = DT / CD`; `CD == DT` (3 dòng) → `CD = DT / MT`. Diện tích giữ nguyên vì là số trên giấy chứng nhận.
- **Hai cột tỷ lệ, chỉ để quan sát:** `ty_le_kt_dt = MT × CD / DT` (thửa cân đối ≈ 1) và `ty_le_dai_rong = CD / MT` (thửa ống có giá trị lớn). Mỗi cột NaN đúng 1 dòng.

### B7 · Khử trùng lặp lần 2

Cùng cách chọn bản giữ như B2, chạy sau khi làm sạch, vì các bước sửa có thể làm hai dòng trở nên giống hệt nhau. Loại **5 dòng**, còn **43.466**.

Ở bước này `Chi tiết` được so ở dạng chữ trước khi tách ý. Nếu so trên bản đã tách, 110 dòng nữa sẽ bị gộp nhầm, vì bộ tách bỏ số nhà nên các lô liền kề như `141` và `143` cùng thành `mt`.

### B8 · Bổ sung ba cột chỉ số

`Chi_so_ngo`, `Chi_so_dientich` và `Chi_so_hinhdang` là các cột công ty **suy ra** từ các cột khác, không phải nhập liệu. Bước này bổ sung lại chúng theo rule của công ty, **tính trên giá trị gốc chưa làm sạch** (độ rộng gốc, diện tích gốc, chuỗi Hình dáng nguyên văn), đúng như cách công ty tính.

**B8.1 · Hai cột chia mức** (tính theo cận trên):

| Cột | Đầu vào | Rule |
|---|---|---|
| `Chi_so_ngo` | độ rộng ngõ/ngách gốc (m) | ≤1 → 1; ≤1,5 → 2; ≤2 → 3; ≤2,5 → 4; ≤3 → 5; ≤3,5 → 6; ≤4 → 7; ≤5 → 8; ≤7 → 9; ≤10 → 10; ≤15 → 11; ≤25 → 12; trên 25 → 13 |
| `Chi_so_dientich` | diện tích gốc (m²) | ≤30 → 1; ≤50 → 2; ≤70 → 3; ≤100 → 4; ≤150 → 5; ≤200 → 6; ≤300 → 7; trên 300 → 8 |

Khớp **100%** cột có sẵn của công ty, trên cả train và test.

**B8.2 · Bảng tra hệ số hình dáng.** Rule của công ty: lấy chuỗi tự do ở cột Hình dáng, tra **nguyên văn** trong sheet Hình dáng; tra được thì lấy hệ số (0,70 – 1,05), không tra được thì gán mặc định **0,95**.

Sheet gốc của công ty không có trong dữ liệu, nên bảng tra được **dựng từ dữ liệu gốc**: với mỗi chuỗi Hình dáng nguyên văn (129 chuỗi), lấy hệ số công ty đã gán nhiều nhất cho chuỗi đó. Bảng của công ty đổi theo thời gian (ví dụ `Chữ L` là 1,03 trong năm 2025 nhưng 0,94 từ cuối 2025), nên ưu tiên các dòng từ **01/01/2026** (64 chuỗi); chuỗi không có dòng nào từ ngày đó thì dùng toàn giai đoạn (65 chuỗi). Bảng giữ nguyên khoảng trắng và ký tự đặc biệt của từng chuỗi, vì rule tra nguyên văn.

Các hệ số chính:

| Nhãn | Hệ số |
|---|---|
| Cân đối | 1,00 |
| Khá cân đối | 0,96 |
| Không cân đối | 0,93 |
| Nở hậu | 1,05 |
| Tóp hậu | 0,95 |
| Hình L | 0,90 |

**B8.3 · Áp hệ số.** Tra chuỗi nguyên văn của từng dòng; không tra được thì gán 0,95. Khớp cột có sẵn của công ty ở **99,69%** dòng train (136 dòng lệch) và **100%** dòng test. 136 dòng lệch là các dòng cũ năm 2025, phần lớn thuộc Q1–Q2/2025, mà công ty áp hệ số theo bảng trước khi cập nhật (ví dụ `Chữ L` 1,03 → 0,94).

Nếu có sheet gốc của công ty, thay `bang_tra_he_so_hinh_dang.csv` bằng sheet đó (giữ hai cột `hinh_dang_nguyen_van` và `he_so`).

### B9 · Kiểm thử và xuất file

Notebook tự kiểm các điều kiện sau trước khi xuất:

- 43.466 dòng, 42 cột; không còn dòng trùng
- Không còn chữ hoa ở bất kỳ cột chữ nào
- VT chỉ còn `vt1`–`vt4`; số mặt tiền chỉ 1–4 hoặc NaN; không còn 0 ở độ rộng và mặt tiền; KC không vượt 2.500
- Không dòng nào `chi_la_mat_tien` mà VT khác `vt1` hoặc `KC > 0`, trừ dòng có `canh_bao_mt_vt234`
- Mọi dòng VT khác `vt1` có `KC = 0` đều có `co_bang_chung_mat_tien`
- `Chi tiết` đúng định dạng `ý; ý`
- `Đường phố` còn đúng 11 NaN; target > 0; cột tỷ lệ không có giá trị vô cực
- `Chi_so_ngo` chỉ nhận 1–13, `Chi_so_dientich` chỉ nhận 1–8, `Chi_so_hinhdang` nằm trong 0,70–1,05; cả ba không NaN

Tổng **900 dòng bị sửa** (2,07%).

---

## Cột đầu ra (42 cột)

| Nhóm | Cột |
|---|---|
| 14 cột đã làm sạch | như B0; `Chi tiết` ở dạng các ý cách nhau bằng `; ` |
| 9 cột giữ nguyên nội dung | `Ghep`, `Don_gia_TB`, `Min`, `Max`, `Độ lệch chuẩn`, `Số lượng mẫu`, `Tháng`, `Quý_Năm`, `Vùng` |
| 2 cột bổ sung | `Thành phố/Quận/Huyện/Thị xã` (40 giá trị), `Mã tài sản` (có ở 7.101 dòng, là các bản ghi TSTĐ) |
| Target | `Đơn giá quyền sử dụng đất (đ/m2)1` |
| Hình dáng sạch | `hinh_dang_sach` (45 nhãn) |
| Ba cột chỉ số | `Chi_so_ngo` (1–13), `Chi_so_dientich` (1–8), `Chi_so_hinhdang` (0,70–1,05) |
| Hai cột tỷ lệ | `ty_le_kt_dt`, `ty_le_dai_rong` |
| Cờ nhận diện | `la_hem`, `co_mat_tien`, `chi_la_mat_tien`, `la_so_nha`, `co_bang_chung_mat_tien` |
| Cờ đã sửa | `da_sua_vt_theo_mat_tien` (161), `kc_thieu_du_lieu` (736), `da_sua_kich_thuoc` (3) |
| Cờ cảnh báo, chưa sửa | `canh_bao_vt1` (425), `canh_bao_mt_vt234` (1.804) |

Số trong ngoặc là số dòng True trong file đầu ra, tức sau B7.

**Ô thiếu trong file đầu ra:** `Đoạn đường` 37.314, `Mã tài sản` 36.365, `Chi tiết` 4.405, `Yếu tố khác` 769, `KC` 737, `Số mặt tiền tiếp giáp` 21, `hinh_dang_sach` 15, `Đường phố` 11, bốn cột `Don_gia_TB`/`Min`/`Max`/`Độ lệch chuẩn` mỗi cột 8, `Độ rộng ngõ/ngách` 3, `Kích thước mặt tiền` 1, hai cột tỷ lệ mỗi cột 1. Ba cột chỉ số không thiếu ô nào.

## Lưu ý khi dùng dữ liệu

- **Ba cột có tên không khớp nội dung** (giữ nguyên tên gốc): `Số lượng mẫu` thực chất là tháng 1–12 của ngày hiệu lực, `Độ lệch chuẩn` là số quan sát trong nhóm `Ghep`, `Tháng` là quý.
- **`Don_gia_TB`, `Min`, `Max`, `Độ lệch chuẩn`** là giá trị của nguồn, được tính khi dữ liệu còn chứa bản sao. Nếu cần khung giá tính theo kiểu loại bỏ chính dòng đang xét (LOO), hãy tính lại từ file đầu ra. Nhóm chỉ có một mẫu có Min = 0,85 × TB và Max = 1,15 × TB.
- **Ba cột chỉ số tính trên giá trị gốc**, nên có thể khác cột đã làm sạch: 3 dòng có độ rộng gốc bằng 0 nhận `Chi_so_ngo = 1` trong khi cột độ rộng sạch là NaN; dòng có Hình dáng gõ sai nhận hệ số mặc định 0,95 như công ty đã áp.
- **`Chi tiết` sau khi tách đã mất số nhà và mô tả nhà/đất.** File đầu ra không có cột số dòng gốc, nên không nối ngược trực tiếp về file gốc được (`Mã tài sản` chỉ có ở các bản ghi TSTĐ).
- **Chia train/validation theo thời gian** dùng `Quý_Năm` và `Số lượng mẫu` (tháng), cả hai đều đầy đủ. Mỗi tài sản chỉ còn một bản (bản muộn nhất), nên không còn rò rỉ giữa các quý do bản sao.

## Đã kiểm và quyết định không làm

- Không coi `MT > CD` hay `MT == CD` là lỗi: thửa rộng ngang và thửa vuông là có thật.
- Không coi `MT == KC` hay `CD == KC` là lỗi: `CD == KC` xuất hiện ít hơn mức ngẫu nhiên, `MT == KC` chỉ nhiều hơn 1,4 lần và phần lớn có kích thước nhất quán.
- Không cắt ngoại lai (`Diện tích`, `MT`, `CD`, độ rộng > 100 m, target > 1 tỷ đ/m²).
- Không dùng ý vị trí trong `Chi tiết` đã tách để sửa VT hay KC.
- Không gom `Hình dáng` thành vài nhóm lớn, không bỏ mức độ.
- Không suy tên đường cho 11 dòng thiếu `Đường phố`.

---

## Tập test `tstd_HCM_3`

Notebook `pipeline_lam_sach_tstd_HCM_3.ipynb` áp cùng quy trình (cùng các hàm trong `lam_sach.ipynb`) cho `data/raw/tstd_HCM_3.xlsx` (710 dòng × 67 cột, toàn bộ TSTĐ, ngày hiệu lực trong tháng 5/2026). Notebook đọc `bang_tra_he_so_hinh_dang.csv` do notebook train tạo ra, và đọc lại `data/raw/train_HCM_3.xlsx` ở bước cuối để đối chiếu Mã tài sản (thêm khoảng 1–2 phút).

- File test không có `Số lượng mẫu` và `Tháng`, nên đầu ra có **693 dòng × 40 cột**.
- Bảng tra hệ số hình dáng đọc từ train, không dựng lại từ test. Ba cột chỉ số khớp cột của công ty ở 100% dòng test.
- Khử trùng lặp loại 17 dòng, trong đó **14 dòng có Mã tài sản khác bản được giữ** (ví dụ 6 lô cùng thông số và cùng giá ở phường thới hòa). Danh sách ở `tstd_dong_bi_loai.csv`.
- 69 Mã tài sản của test đã có trong train từ lần định giá trước (23 trường hợp cùng đơn giá). Danh sách ở `tstd_tai_san_co_trong_train.csv`. Không dòng test nào giống hệt một dòng train.
