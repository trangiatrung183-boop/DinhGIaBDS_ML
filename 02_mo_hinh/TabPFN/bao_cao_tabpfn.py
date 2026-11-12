"""Đọc checkpoint của chay_tabpfn.py + dự đoán LightGBM/HGB đã có ở 02_mo_hinh/checkpoints, xuất:
    ket_qua_tabpfn.xlsx, predictions_val.csv, predictions_test.csv, predictions_ngoai_thoi_gian.csv,
    bd_tabpfn_so_sanh.png, bang_so_lieu.md

    python 02_mo_hinh/TabPFN/bao_cao_tabpfn.py
"""
import sys, json
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
SS = HERE.parent
sys.path.insert(0, str(SS))
from a1_common import *   # noqa: F401,F403

CK, CK_SS = HERE / "checkpoints", SS / "checkpoints"
L = lambda p, n: joblib.load(p / f"{n}.joblib")
QUAN = "Thành phố/Quận/Huyện/Thị xã"
N_BOOT = 5000

train_raw = load_clean(TRAIN_XLS); test_raw = load_clean(TEST_XLS)
y_all = train_raw[TARGET].astype(float).values; y_test = test_raw[TARGET].astype(float).values
tr_idx, va_idx = train_test_split(np.arange(len(train_raw)), test_size=TEST_SIZE, random_state=SEED)
va_raw = train_raw.iloc[va_idx]
Y = {"Val": y_all[va_idx], "Test": y_test}
REG = {"Val": va_raw[VUNG].str.upper().values, "Test": test_raw[VUNG].str.upper().values}

main, oot = L(CK, "main"), L(CK, "oot")
VS = main["va_sel"]                                   # val: mẫu cố định VAL_SUB dòng (cùng mẫu cho mọi mô hình)
va_raw = va_raw.iloc[VS]; Y["Val"] = Y["Val"][VS]; REG["Val"] = REG["Val"][VS]
sub = lambda d: {"Val": d["Val"][VS], "Test": d["Test"]}
pilot = L(CK, "pilot") if (CK / "pilot.joblib").exists() else None
LGB, HGB = L(CK_SS, "m_lightgbm"), L(CK_SS, "m_hgb")
OL, OO = L(CK_SS, "oot_lightgbm"), L(CK_SS, "oot_other")
N_SUB = len(main["res"]); N_CTX = main["res"][0]["n_ctx"]

TAB = f"TabPFN v2 ({N_SUB}×{N_CTX:,} dòng)".replace(",", ".")
TAB1 = f"TabPFN v2 (1×{N_CTX:,} dòng)".replace(",", ".")
REF, HG = "LightGBM (A1)", "HistGradientBoosting"
P = {REF: sub(LGB["pred"]), HG: sub(HGB["pred"]),
     TAB: {"Val": main["pred"][0], "Test": main["pred"][1]},
     TAB1: {"Val": main["pred_each"][0][0], "Test": main["pred_each"][1][0]}}
if pilot is not None:
    P["TabPFN v2 (1×2.000 dòng, thử)"] = sub({"Val": pilot["pred"][0], "Test": pilot["pred"][1]})
MODELS = list(P)
sheets = {}

# ---------------- 1. cấu hình ----------------
sheets["1_cau_hinh"] = pd.DataFrame([
    {"mục": "mô hình", "giá trị": "TabPFN v2 regressor (Prior-Labs/TabPFN-v2-reg), gói tabpfn 9.2.0, GPU RTX 3050 4 GB"},
    {"mục": "vì sao v2", "giá trị": "v2.5/v2.6/v3/v3.5 cần đăng nhập Prior Labs để chấp nhận giấy phép trước khi tải trọng số"},
    {"mục": "cách chia", "giá trị": "giống A1: SEED 42, val 20% của train (train 34.772 · val 8.694), test = tstd 693"},
    {"mục": "val dùng để đánh giá", "giá trị": f"mẫu ngẫu nhiên cố định {len(VS)}/8.694 dòng (SEED 42) để giới hạn thời gian; "
                                             "LightGBM/HGB cũng tính trên đúng mẫu này. Test dùng đủ 693 dòng"},
    {"mục": "target", "giá trị": "log1p(đơn giá); dự đoán = expm1(trung bình log qua các tập con)"},
    {"mục": "ngữ cảnh", "giá trị": f"{N_SUB} tập con ngẫu nhiên, mỗi tập {N_CTX} dòng lấy từ train 34.772 (không lặp trong một tập)"},
    {"mục": "n_estimators (ensemble nội bộ TabPFN)", "giá trị": 8},
    {"mục": "số đặc trưng", "giá trị": main["n_feat"]},
    {"mục": "đặc trưng", "giá trị": "; ".join(main["features"])},
    {"mục": "mã hoá Phường/xã, Đường phố", "giá trị": "target encoding log1p làm mượt k=20 (train: out-of-fold 5 phần) + số lần xuất hiện"},
    {"mục": "mã hoá Vùng, VT, Lợi thế KD, Hình dáng", "giá trị": "mã thứ tự, khai báo là biến phân loại cho TabPFN"},
    {"mục": "Quý_Năm", "giá trị": "đổi thành số: năm + (quý−1)/4"},
    {"mục": "không dùng", "giá trị": "600 cột TF-IDF 'Yếu tố khác' (TabPFN v2 giới hạn 500 đặc trưng)"},
    {"mục": "chọn siêu tham số", "giá trị": "không — cấu hình cố định, không nhìn test"},
])

# ---------------- 2. tài nguyên ----------------
rows = []
for lab, r in ([("thử 2.000 dòng", pilot)] if pilot is not None else []) + [("chính", main), ("ngoài thời gian", oot)]:
    for i, s in enumerate(r["res"]):
        rows.append({"lượt": lab, "tập con": i + 1, **s,
                     "giây / 1.000 dòng dự đoán": round(s["secs"] / s["n_pred"] * 1000, 2)})
tn = pd.DataFrame(rows)
tong = tn.groupby("lượt", sort=False).agg(**{"tổng giây": ("secs", "sum"), "VRAM đỉnh (GB)": ("vram_peak_gb", "max"),
                                             "RAM tiến trình tối đa (GB)": ("ram_gb", "max")}).reset_index()
so = pd.DataFrame([{"mô hình": REF, "giây (huấn luyện + dự đoán, cấu hình chọn)": LGB.get("secs")},
                   {"mô hình": HG, "giây (huấn luyện + dự đoán, cấu hình chọn)": HGB.get("secs")},
                   {"mô hình": TAB, "giây (huấn luyện + dự đoán, cấu hình chọn)": tn[tn["lượt"] == "chính"]["secs"].sum()}])
sheets["2_tai_nguyen"] = (tn, tong, so)

# ---------------- 3. đánh giá Val/Test theo vùng ----------------
rows = []
for m in MODELS:
    for k in ["Val", "Test"]:
        rows.append({"mô hình": m, "tập": k, "vùng": "Tất cả", **metrics_full(Y[k], P[m][k])})
        for v in ["HCM1", "HCM2"]:
            s = REG[k] == v
            rows.append({"mô hình": m, "tập": k, "vùng": v, **metrics_full(Y[k][s], P[m][k][s])})
sheets["3_danh_gia"] = pd.DataFrame(rows)

# ---------------- 4. mã tài sản có trong train / mới ----------------
known = set(pd.read_csv(ROOT / "data" / "processed" / "tstd_tai_san_co_trong_train.csv", dtype=str)["ma_tai_san"].str.strip())
in_tr = test_raw["Mã tài sản"].astype(str).str.strip().isin(known).values
sheets["4_test_ma_ts"] = pd.DataFrame([{"mô hình": m, "nhóm": lab, **metrics_full(y_test[s], P[m]["Test"][s])}
                                       for m in MODELS for lab, s in [("Có trong train", in_tr), ("Mã mới", ~in_tr)]])

# ---------------- 5. bootstrap theo cặp (cùng chỉ số mẫu với phan_tich.py) ----------------
rng = np.random.default_rng(SEED)
IDX = rng.integers(0, len(y_test), size=(N_BOOT, len(y_test)))
APE = {m: np.abs(P[m]["Test"] - y_test) / y_test * 100 for m in MODELS}
STAT = {"MAPE": (lambda a: a.mean(), lambda A: A.mean(axis=1), -1),
        "APE<10%": (lambda a: (a < 10).mean() * 100, lambda A: (A < 10).mean(axis=1) * 100, +1),
        "APE<15%": (lambda a: (a < 15).mean() * 100, lambda A: (A < 15).mean(axis=1) * 100, +1)}
BSS = {s: {m: fb(APE[m][IDX]) for m in MODELS} for s, (_, fb, _) in STAT.items()}
rows = []
for s, (f, _, sign) in STAT.items():
    for m in MODELS:
        lo, hi = np.percentile(BSS[s][m], [2.5, 97.5])
        row = {"chỉ số": s, "mô hình": m, "giá trị Test": f(APE[m]), "KTC95 thấp": lo, "KTC95 cao": hi}
        for ref in [REF, HG]:
            if m == ref:
                continue
            d = BSS[s][m] - BSS[s][ref]; dlo, dhi = np.percentile(d, [2.5, 97.5])
            if (dlo > 0 and sign > 0) or (dhi < 0 and sign < 0): kl = f"tốt hơn {ref}"
            elif (dlo > 0 and sign < 0) or (dhi < 0 and sign > 0): kl = f"kém hơn {ref}"
            else: kl = "không phân biệt được"
            tag = "LGB" if ref == REF else "HGB"
            row.update({f"Δ so với {tag}": f(APE[m]) - f(APE[ref]), f"Δ {tag} KTC95": f"[{dlo:+.2f}; {dhi:+.2f}]",
                        f"kết luận so với {tag}": kl})
        rows.append(row)
BOOT = pd.DataFrame(rows)
sheets["5_bootstrap"] = BOOT

# ---------------- 6. số tập con ngữ cảnh ----------------
rows = []
for n in range(1, N_SUB + 1):
    for j, k in enumerate(["Val", "Test"]):
        p = np.expm1(np.mean([np.log1p(main["pred_each"][j][i]) for i in range(n)], axis=0))
        rows.append({"số tập con": n, "dòng ngữ cảnh khác nhau (ước)": None, "tập": k, **metrics_full(Y[k], p)})
sheets["6_so_tap_con"] = pd.DataFrame(rows).drop(columns="dòng ngữ cảnh khác nhau (ước)")

# ---------------- 7. ngoài thời gian ----------------
rows = []
for m, pin, pout in [(REF, P[REF]["Test"], OL["pred"]["Test"]), (HG, P[HG]["Test"], OO["pred"]["Test"]),
                     (TAB, P[TAB]["Test"], oot["pred"][0])]:
    for cach, p in [("Gốc: train có q2-2026", pin), ("Ngoài thời gian: train bỏ q2-2026", pout)]:
        for v in ["Tất cả", "HCM1", "HCM2"]:
            s = np.ones(len(y_test), bool) if v == "Tất cả" else REG["Test"] == v
            rows.append({"mô hình": m, "cách huấn luyện": cach, "vùng": v, **metrics_full(y_test[s], p[s])})
sheets["7_ngoai_thoi_gian"] = pd.DataFrame(rows)
assert OO["model_name"] == HG

# ---------------- 8. theo phân vị đơn giá & quận ----------------
q = pd.qcut(y_test, 4, labels=["Q1 (thấp nhất)", "Q2", "Q3", "Q4 (cao nhất)"])
rows = []
for lab in q.categories:
    s = np.asarray(q == lab)
    for m in [REF, HG, TAB]:
        rows.append({"nhóm theo": "phân vị đơn giá", "nhóm": lab, "mô hình": m, **metrics_full(y_test[s], P[m]["Test"][s])})
for d, g in pd.Series(np.arange(len(y_test))).groupby(test_raw[QUAN].fillna("(trống)").values):
    s = g.values
    for m in [REF, HG, TAB]:
        rows.append({"nhóm theo": "quận/huyện", "nhóm": d, "mô hình": m, **metrics_full(y_test[s], P[m]["Test"][s])})
grp = pd.DataFrame(rows)
grp["cảnh báo"] = np.where(grp["n"] < 10, "n<10: không đủ tin cậy", "")
sheets["8_phan_vi_quan"] = grp

with pd.ExcelWriter(HERE / "ket_qua_tabpfn.xlsx") as w:
    for name, obj in sheets.items():
        if isinstance(obj, tuple):
            r0 = 0
            for t in obj:
                t.to_excel(w, sheet_name=name, index=False, startrow=r0); r0 += len(t) + 3
        else:
            obj.to_excel(w, sheet_name=name, index=False)

# ---------------- predictions ----------------
def base(raw):
    return pd.DataFrame({"Mã tài sản": raw["Mã tài sản"].values, "Vùng": raw[VUNG].values,
                         QUAN: raw[QUAN].values, "Quý_Năm": raw["Quý_Năm"].values})
for k, raw in [("Val", va_raw), ("Test", test_raw)]:
    t = base(raw); t["Thực tế"] = Y[k]
    for m in [REF, HG, TAB, TAB1]:
        t[f"Dự đoán · {m}"] = P[m][k]
    t[f"APE % · {TAB}"] = np.abs(P[TAB][k] - Y[k]) / Y[k] * 100
    t.to_csv(HERE / f"predictions_{k.lower()}.csv", index=False, encoding="utf-8-sig")
t = base(test_raw); t["Thực tế"] = y_test
t["Dự đoán · LightGBM ngoài thời gian"] = OL["pred"]["Test"]
t["Dự đoán · HistGradientBoosting ngoài thời gian"] = OO["pred"]["Test"]
t[f"Dự đoán · {TAB} ngoài thời gian"] = oot["pred"][0]
t.to_csv(HERE / "predictions_ngoai_thoi_gian.csv", index=False, encoding="utf-8-sig")

# ---------------- biểu đồ: 3 chỉ số chính, chấm + KTC95 ----------------
SHOW = [REF, HG, TAB]
COL = {REF: "#2a78d6", HG: "#eda100", TAB: "#008300"}      # LGB/HGB giữ màu cũ của bd1; TabPFN = slot 6
fig, axes = plt.subplots(1, 3, figsize=(13, 3.6), constrained_layout=True)
for ax, s in zip(axes, STAT):
    for i, m in enumerate(SHOW[::-1]):
        r = BOOT[(BOOT["chỉ số"] == s) & (BOOT["mô hình"] == m)].iloc[0]
        ax.plot([r["KTC95 thấp"], r["KTC95 cao"]], [i, i], color=COL[m], lw=2, solid_capstyle="round")
        ax.plot(r["giá trị Test"], i, "o", ms=9, color=COL[m], mec="white", mew=2)
        ax.annotate(f"{r['giá trị Test']:.1f}", (r["giá trị Test"], i), xytext=(0, 9), textcoords="offset points",
                    ha="center", fontsize=9, color="#3d3d3a")
    ax.set_yticks(range(len(SHOW))); ax.set_yticklabels(SHOW[::-1] if ax is axes[0] else [""] * len(SHOW))
    ax.set_title(s + (" (thấp là tốt)" if s == "MAPE" else " (cao là tốt)"), fontsize=10, color="#1f1f1d")
    ax.set_ylim(-0.6, len(SHOW) - 0.4)
    ax.grid(axis="x", color="#e6e5e0", lw=0.8); ax.set_axisbelow(True)
    for sp in ["top", "right", "left"]:
        ax.spines[sp].set_visible(False)
    ax.spines["bottom"].set_color("#c3c2b7"); ax.tick_params(colors="#5f5e5a", length=0)
    ax.set_xlabel("%", color="#5f5e5a")
fig.suptitle(f"Tập test tstd (n = {len(y_test)}) — chấm: giá trị, vạch: KTC 95% bootstrap theo cặp ({N_BOOT} lần)",
             fontsize=10, color="#1f1f1d")
fig.savefig(HERE / "bd_tabpfn_so_sanh.png", dpi=150, facecolor="white")

# ---------------- bảng số liệu nhanh (markdown) ----------------
def fmt(x, d=2):
    return f"{x:.{d}f}".replace(".", ",")
lines = ["| Mô hình | MAPE % [KTC] | APE<10% [KTC] | APE<15% [KTC] | Δ MAPE vs LGB | kết luận MAPE vs LGB |", "|---|---|---|---|---|---|"]
for m in MODELS:
    c = {s: BOOT[(BOOT["chỉ số"] == s) & (BOOT["mô hình"] == m)].iloc[0] for s in STAT}
    lines.append(f"| {m} | " + " | ".join(f"{fmt(c[s]['giá trị Test'])} [{fmt(c[s]['KTC95 thấp'])}–{fmt(c[s]['KTC95 cao'])}]" for s in STAT)
                 + f" | {c['MAPE'].get('Δ LGB KTC95', '—') if m != REF else '—'} | {c['MAPE'].get('kết luận so với LGB', '—') if m != REF else '—'} |")
(HERE / "bang_so_lieu.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
print("\n".join(lines))
print(sheets["3_danh_gia"].query("vùng=='Tất cả'")[["mô hình", "tập", "R2", "MAPE", "MdAPE", "MPE (dự đoán−thực, %)", "APE<10%", "APE<15%"]].round(2).to_string())
print(sheets["6_so_tap_con"][["số tập con", "tập", "MAPE", "APE<10%", "APE<15%"]].round(2).to_string())
print(sheets["7_ngoai_thoi_gian"].query("vùng=='Tất cả'")[["mô hình", "cách huấn luyện", "MAPE", "MPE (dự đoán−thực, %)", "APE<10%", "APE<15%"]].round(2).to_string())
print(BOOT.round(2).to_string())
print(tong.to_string())
