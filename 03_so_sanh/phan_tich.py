"""Bước 4, 5, 6, 7 (bảng), 8 (bảng), 9, 10 + xuất ket_qua.xlsx, predictions_*.csv, 4 biểu đồ PNG.

    python 03_so_sanh/phan_tich.py        (chạy sau chay_mo_hinh.py; chỉ đọc checkpoint, không huấn luyện lại)
"""
import sys, json
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
import lightgbm as lgb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "02_mo_hinh"))
from a1_common import *   # noqa: F401,F403

HERE = Path(__file__).resolve().parent
CK = HERE.parent / "02_mo_hinh" / "checkpoints"
L = lambda n: joblib.load(CK / f"{n}.joblib")
QUAN = "Thành phố/Quận/Huyện/Thị xã"
N_BOOT = 5000

# ---------------- dữ liệu & dự đoán ----------------
train_raw = load_clean(TRAIN_XLS); test_raw = load_clean(TEST_XLS)
y_all = train_raw[TARGET].astype(float).values; y_test = test_raw[TARGET].astype(float).values
tr_idx, va_idx = train_test_split(np.arange(len(train_raw)), test_size=TEST_SIZE, random_state=SEED)
tr_raw, va_raw = train_raw.iloc[tr_idx], train_raw.iloc[va_idx]
Y = {"Train": y_all[tr_idx], "Val": y_all[va_idx], "Test": y_test}
REG = {"Train": tr_raw[VUNG].str.upper().values, "Val": va_raw[VUNG].str.upper().values,
       "Test": test_raw[VUNG].str.upper().values}

R = {"LightGBM (A1)": L("m_lightgbm"), "Hedonic OLS": L("m_hedonic"), "Random Forest": L("m_rf"),
     "HistGradientBoosting": L("m_hgb"), "CatBoost": L("m_catboost")}
MODELS = list(R)
P = {m: R[m]["pred"] for m in MODELS}
REF = "LightGBM (A1)"
COLORS = dict(zip(MODELS, ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]))

sheets = {}

# ---------------- Bước 3 (phụ): siêu tham số & hedonic ----------------
hp = []
for m in MODELS:
    r = R[m]
    hp.append({"mô hình": m, "tham số chọn": json.dumps(r["params"], ensure_ascii=False, default=str),
               "best_iter": r.get("best_iter"), "MAPE Val": metrics_full(Y["Val"], P[m]["Val"])["MAPE"],
               "giây (cấu hình chọn)": r.get("secs")})
srch = pd.concat([R[m]["search_log"] for m in MODELS if "search_log" in R[m]], ignore_index=True)
sheets["3_sieu_tham_so"] = (pd.DataFrame(hp), srch)
h = R["Hedonic OLS"]
sheets["3_hedonic_he_so"] = h["coef"]
sheets["3_hedonic_ghi_chu"] = pd.DataFrame(
    [{"mục": "đặc tả", "giá trị": h["params"]["spec"]},
     {"mục": "n, k", "giá trị": f"{h['n']}, {h['k']}"},
     {"mục": "R2 (thang log, train)", "giá trị": round(h["R2_log_train"], 4)},
     {"mục": "cột bị bỏ do cộng tuyến hoàn hảo", "giá trị": "; ".join(h["dropped"]) or "(không)"},
     {"mục": "mức tham chiếu", "giá trị": json.dumps(h["ref_levels"], ensure_ascii=False)},
     {"mục": "dòng test có mức chưa gặp trong train (gán = tham chiếu)", "giá trị": json.dumps(h["unseen_test"], ensure_ascii=False)}])

# ---------------- Bước 4: Train/Val/Test, tổng + theo vùng ----------------
rows = []
for m in MODELS:
    for k in ["Train", "Val", "Test"]:
        rows.append({"mô hình": m, "tập": k, "vùng": "Tất cả", **metrics_full(Y[k], P[m][k])})
        for v in ["HCM1", "HCM2"]:
            s = REG[k] == v
            rows.append({"mô hình": m, "tập": k, "vùng": v, **metrics_full(Y[k][s], P[m][k][s])})
met = pd.DataFrame(rows)
sheets["4_danh_gia"] = met

# ---------------- Bước 5: mã tài sản có trong train vs mới ----------------
known = set(pd.read_csv(ROOT / "data" / "processed" / "tstd_tai_san_co_trong_train.csv", dtype=str)["ma_tai_san"].str.strip())
in_tr = test_raw["Mã tài sản"].astype(str).str.strip().isin(known).values
rows = []
for m in MODELS:
    for lab, s in [("Có trong train", in_tr), ("Mã mới", ~in_tr)]:
        rows.append({"mô hình": m, "nhóm": lab, **metrics_full(y_test[s], P[m]["Test"][s])})
sheets["5_test_ma_ts"] = pd.DataFrame(rows)

# ---------------- Bước 6: bootstrap theo cặp ----------------
rng = np.random.default_rng(SEED)
IDX = rng.integers(0, len(y_test), size=(N_BOOT, len(y_test)))      # dùng CHUNG cho mọi mô hình
APE = {m: np.abs(P[m]["Test"] - y_test) / y_test * 100 for m in MODELS}
BS = {m: APE[m][IDX].mean(axis=1) for m in MODELS}
# Chỉ số chính: MAPE (thấp là tốt), APE<10%, APE<15% (cao là tốt)
STAT = {"MAPE": (lambda a: a.mean(), lambda A: A.mean(axis=1), -1),
        "APE<10%": (lambda a: (a < 10).mean() * 100, lambda A: (A < 10).mean(axis=1) * 100, +1),
        "APE<15%": (lambda a: (a < 15).mean() * 100, lambda A: (A < 15).mean(axis=1) * 100, +1)}
BSS = {s: {m: f_b(APE[m][IDX]) for m in MODELS} for s, (_, f_b, _) in STAT.items()}
rows = []
for s, (f, _, sign) in STAT.items():
    for m in MODELS:
        lo, hi = np.percentile(BSS[s][m], [2.5, 97.5])
        d = BSS[s][m] - BSS[s][REF]; dlo, dhi = np.percentile(d, [2.5, 97.5])
        if m == REF: kl = "—"
        elif (dlo > 0 and sign > 0) or (dhi < 0 and sign < 0): kl = "tốt hơn LightGBM"
        elif (dlo > 0 and sign < 0) or (dhi < 0 and sign > 0): kl = "kém hơn LightGBM"
        else: kl = "không phân biệt được"
        rows.append({"chỉ số": s, "mô hình": m, "giá trị Test": f(APE[m]), "KTC95 thấp": lo, "KTC95 cao": hi,
                     "Δ so với LightGBM": f(APE[m]) - f(APE[REF]),
                     "Δ KTC95 thấp": dlo if m != REF else np.nan, "Δ KTC95 cao": dhi if m != REF else np.nan,
                     "kết luận": kl})
boot_all = pd.DataFrame(rows)
boot = boot_all[boot_all["chỉ số"] == "MAPE"].rename(columns={"giá trị Test": "MAPE Test"})
sheets["6_bootstrap"] = boot_all

# ---------------- Bước 7: ngoài thời gian ----------------
ol, oo = L("oot_lightgbm"), L("oot_other")
rows = []
for lab, mname, pin, pout in [("LightGBM (A1)", REF, P[REF]["Test"], ol["pred"]["Test"]),
                              (oo["model_name"], oo["model_name"], P[oo["model_name"]]["Test"], oo["pred"]["Test"])]:
    for cach, p in [("Gốc: train có q2-2026 (chia ngẫu nhiên)", pin), ("Ngoài thời gian: train bỏ q2-2026", pout)]:
        for v in ["Tất cả", "HCM1", "HCM2"]:
            s = np.ones(len(y_test), bool) if v == "Tất cả" else REG["Test"] == v
            rows.append({"mô hình": lab, "cách huấn luyện": cach, "vùng": v, **metrics_full(y_test[s], p[s])})
oot = pd.DataFrame(rows)
oot_note = pd.DataFrame([{"mục": "dòng train dùng (≠ q2-2026)", "giá trị": ol["n_train"] + ol["n_val"]},
                         {"mục": "chia train/val", "giá trị": f"{ol['n_train']}/{ol['n_val']} (seed 42, 0.2)"},
                         {"mục": "best_iter LightGBM ngoài thời gian", "giá trị": ol["best_iter"]},
                         {"mục": "mô hình còn lại (chọn theo Val MAPE)", "giá trị": oo["model_name"]},
                         {"mục": "tham số (giữ nguyên từ bước 3)", "giá trị": json.dumps(oo["params"], default=str)},
                         {"mục": "lưu ý", "giá trị": "Quý_Năm=q2-2026 của tstd là mức chưa gặp khi huấn luyện ngoài thời gian"}])
sheets["7_ngoai_thoi_gian"] = (oot, oot_note)

# ---------------- Bước 8: bỏ nhóm biến ----------------
ABL = {"Đầy đủ (A1)": None, "a) bỏ Chi tiết + Yếu tố khác": "a_bo_ChiTiet_YeuToKhac",
       "b) bỏ Đường phố": "b_bo_DuongPho", "c) bỏ Chi tiết + Yếu tố khác + Đường phố": "c_bo_ChiTiet_YeuToKhac_DuongPho"}
rows, ABLP = [], {}
for lab, key in ABL.items():
    r = R[REF] if key is None else L(f"abl_{key}")
    ABLP[lab] = r["pred"]
    a_t = np.abs(r["pred"]["Test"] - y_test) / y_test * 100
    d = a_t[IDX].mean(axis=1) - BS[REF]
    d10 = (a_t[IDX] < 10).mean(axis=1) * 100 - BSS["APE<10%"][REF]
    d15 = (a_t[IDX] < 15).mean(axis=1) * 100 - BSS["APE<15%"][REF]
    for k in ["Val", "Test"]:
        row = {"cấu hình": lab, "số đặc trưng": r.get("n_features", len(L("features_main")[0].columns)),
               "best_iter": r["best_iter"], "tập": k, **metrics_full(Y[k], r["pred"][k])}
        if k == "Test" and key is not None:
            row.update({"Δ MAPE Test so với đầy đủ": a_t.mean() - APE[REF].mean(),
                        "Δ KTC95 thấp": np.percentile(d, 2.5), "Δ KTC95 cao": np.percentile(d, 97.5),
                        "Δ APE<10% so với đầy đủ": (a_t < 10).mean() * 100 - (APE[REF] < 10).mean() * 100,
                        "Δ APE<10% KTC95": f"[{np.percentile(d10, 2.5):+.2f}; {np.percentile(d10, 97.5):+.2f}]",
                        "Δ APE<15% so với đầy đủ": (a_t < 15).mean() * 100 - (APE[REF] < 15).mean() * 100,
                        "Δ APE<15% KTC95": f"[{np.percentile(d15, 2.5):+.2f}; {np.percentile(d15, 97.5):+.2f}]"})
        rows.append(row)
sheets["8_bo_bien"] = pd.DataFrame(rows)

# ---------------- Bước 9: SHAP (TreeSHAP của LightGBM) ----------------
X_tr, X_va, X_te, CATS = L("features_main")
bst = lgb.Booster(model_file=str(CK / "models" / "lightgbm_A1.txt"))
Xc = to_cat(X_te, CATS)
contrib = bst.predict(Xc, pred_contrib=True)            # n × (F+1), cột cuối = giá trị kỳ vọng
add_err = np.abs(contrib.sum(1) - np.log1p(P[REF]["Test"])).max()
F = list(X_te.columns)
GROUPS = {"Vùng": [VUNG], "Phường/xã": [XA], "Đường phố": ["Đường phố"],
          "Độ rộng ngõ": ["Độ rộng ngõ/ngách nhỏ nhất (Từ đường chính đến BĐS)"],
          "Diện tích": ["Diện tích (m2)"], "Mặt tiền": ["Kích thước mặt tiền (m)", "Số mặt tiền tiếp giáp"],
          "Chiều dài": ["Kích thước chiều dài"], "Khoảng cách": ["Khoảng cách đến đường chính (m)"],
          "Lợi thế KD": ["Lợi thế kinh doanh"], "VT": ["Vị trí trong khung giá"], "Quý": ["Quý_Năm"],
          "Hình dáng": ["hinh_dang_sach"],
          "Chi tiết": [c for c in F if c.startswith("ct_")],
          "TF-IDF (Yếu tố khác)": [c for c in F if c.startswith("tfidf_")]}
assert sum(len(v) for v in GROUPS.values()) == len(F)
rows = []
for g, cols in GROUPS.items():
    s = contrib[:, [F.index(c) for c in cols]].sum(1)      # cộng SHAP trong nhóm rồi mới lấy |·|
    rows.append({"nhóm biến": g, "số cột": len(cols), "mean |SHAP| (log)": np.abs(s).mean(),
                 "mean SHAP (log)": s.mean(),
                 "mean |SHAP| HCM1": np.abs(s[REG["Test"] == "HCM1"]).mean(),
                 "mean |SHAP| HCM2": np.abs(s[REG["Test"] == "HCM2"]).mean()})
shap_df = pd.DataFrame(rows).sort_values("mean |SHAP| (log)", ascending=False)
shap_df["tỷ trọng %"] = shap_df["mean |SHAP| (log)"] / shap_df["mean |SHAP| (log)"].sum() * 100
shap_note = pd.DataFrame([{"mục": "phương pháp", "giá trị": "LightGBM Booster.predict(pred_contrib=True) = TreeSHAP (path-dependent), "
                                                         "cùng thuật toán shap.TreeExplainer dùng cho LightGBM; gói shap chưa cài trên máy"},
                          {"mục": "thang", "giá trị": "log1p(đơn giá)"},
                          {"mục": "kiểm tra cộng tính: max|Σshap − log dự đoán|", "giá trị": f"{add_err:.2e}"},
                          {"mục": "giá trị kỳ vọng (log)", "giá trị": float(contrib[0, -1])},
                          {"mục": "ghi chú", "giá trị": "'Mặt tiền' gồm Kích thước mặt tiền + Số mặt tiền tiếp giáp; "
                                                      "thêm nhóm 'Hình dáng' (hinh_dang_sach) vì A1 có dùng"}])
sheets["9_shap"] = (shap_df, shap_note)

# ---------------- Bước 10: sai số LightGBM theo phân vị & quận ----------------
pL = P[REF]["Test"]
q = pd.qcut(y_test, 4, labels=["Q1 (thấp nhất)", "Q2", "Q3", "Q4 (cao nhất)"])
edges = np.quantile(y_test, [0, .25, .5, .75, 1])
rows = []
for i, lab in enumerate(q.categories):
    s = (q == lab)
    rows.append({"nhóm theo": "phân vị đơn giá thực (test)", "nhóm": lab,
                 "khoảng đơn giá (đ/m²)": f"{edges[i]:,.0f} – {edges[i+1]:,.0f}", **metrics_full(y_test[s], pL[s])})
for d, g in pd.Series(np.arange(len(y_test))).groupby(test_raw[QUAN].fillna("(trống)").values):
    s = g.values
    rows.append({"nhóm theo": "quận/huyện", "nhóm": d, "khoảng đơn giá (đ/m²)": "", **metrics_full(y_test[s], pL[s])})
err = pd.DataFrame(rows)
err["cảnh báo"] = np.where(err["n"] < 10, "n<10: không đủ tin cậy", "")
q_part = err[err["nhóm theo"] != "quận/huyện"]
d_part = err[err["nhóm theo"] == "quận/huyện"].sort_values("n", ascending=False)
sheets["10_sai_so_LGB"] = pd.concat([q_part, d_part], ignore_index=True)

# ---------------- Xuất Excel ----------------
with pd.ExcelWriter(HERE / "ket_qua.xlsx") as w:
    for name, obj in sheets.items():
        if isinstance(obj, tuple):
            r0 = 0
            for t in obj:
                t.to_excel(w, sheet_name=name, index=False, startrow=r0); r0 += len(t) + 3
        else:
            obj.to_excel(w, sheet_name=name, index=False)

# ---------------- predictions_*.csv ----------------
def pred_table(raw, k):
    out = pd.DataFrame({"Mã tài sản": raw["Mã tài sản"].values, "Vùng": raw[VUNG].values,
                        QUAN: raw[QUAN].values, "Quý_Năm": raw["Quý_Năm"].values, "Thực tế": Y[k]})
    for m in MODELS:
        out[f"Dự đoán · {m}"] = P[m][k]
    return out
t = pred_table(test_raw, "Test"); t.insert(1, "Mã có trong train", in_tr)
t.to_csv(HERE / "predictions" / "predictions_test.csv", index=False, encoding="utf-8-sig")
pred_table(va_raw, "Val").to_csv(HERE / "predictions" / "predictions_val.csv", index=False, encoding="utf-8-sig")
pred_table(tr_raw, "Train").to_csv(HERE / "predictions" / "predictions_train.csv", index=False, encoding="utf-8-sig")
po = test_raw[["Mã tài sản", VUNG, QUAN]].copy(); po["Thực tế"] = y_test
po["Dự đoán · LightGBM ngoài thời gian"] = ol["pred"]["Test"]
po[f"Dự đoán · {oo['model_name']} ngoài thời gian"] = oo["pred"]["Test"]
po.to_csv(HERE / "predictions" / "predictions_ngoai_thoi_gian.csv", index=False, encoding="utf-8-sig")
pa = test_raw[["Mã tài sản", VUNG, QUAN]].copy(); pa["Thực tế"] = y_test
for lab, p in ABLP.items():
    pa[f"Dự đoán · {lab}"] = p["Test"]
pa.to_csv(HERE / "predictions" / "predictions_bo_bien.csv", index=False, encoding="utf-8-sig")

# ---------------- Biểu đồ ----------------
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.edgecolor": GRID,
                     "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
                     "axes.titlecolor": INK, "axes.titlesize": 11, "figure.facecolor": "#fcfcfb",
                     "axes.facecolor": "#fcfcfb", "axes.spines.top": False, "axes.spines.right": False})

# 1. so sánh mô hình: 3 chỉ số chính, cùng thứ tự mô hình (theo MAPE)
order = boot.sort_values("giá trị Test" if "giá trị Test" in boot else "MAPE Test", ascending=False)["mô hình"].tolist()
fig, axes = plt.subplots(1, 3, figsize=(15, 3.9), sharey=True)
for ax, (s, xl) in zip(axes, [("MAPE", "MAPE (%) — thấp hơn là tốt"),
                              ("APE<10%", "Tỷ lệ dòng APE < 10% (%) — cao hơn là tốt"),
                              ("APE<15%", "Tỷ lệ dòng APE < 15% (%) — cao hơn là tốt")]):
    bb = boot_all[boot_all["chỉ số"] == s].set_index("mô hình").loc[order]
    for i, (m, r) in enumerate(bb.iterrows()):
        ax.plot([r["KTC95 thấp"], r["KTC95 cao"]], [i, i], color=COLORS[m], lw=2, solid_capstyle="round")
        ax.plot(r["giá trị Test"], i, "o", ms=9, color=COLORS[m], mec="#fcfcfb", mew=2)
        ax.text(r["giá trị Test"], i + 0.28, f"{r['giá trị Test']:.1f}  [{r['KTC95 thấp']:.1f}–{r['KTC95 cao']:.1f}]",
                ha="center", va="bottom", color=INK, fontsize=8.5)
    ax.set_xlabel(xl); ax.set_title(s, loc="left")
    ax.grid(axis="x", color=GRID, lw=0.8); ax.set_axisbelow(True)
    pad = (bb["KTC95 cao"].max() - bb["KTC95 thấp"].min()) * 0.15
    ax.set_xlim(bb["KTC95 thấp"].min() - pad, bb["KTC95 cao"].max() + pad)
    ax.set_ylim(-0.5, len(order) - 0.3)
axes[0].set_yticks(range(len(order))); axes[0].set_yticklabels(order)
fig.suptitle(f"So sánh 5 mô hình trên tập test (n = {len(y_test)}) — chấm: giá trị, vạch: KTC 95% bootstrap (5000 lần)",
             x=0.01, ha="left", color=INK)
fig.tight_layout(); fig.savefig(HERE / "bieu_do" / "bd1_so_sanh_mo_hinh.png", dpi=200); plt.close(fig)

# 2. thực tế – dự đoán
fig, axes = plt.subplots(1, 5, figsize=(17, 3.9), sharex=True, sharey=True)
lo_, hi_ = y_test.min() * 0.7, y_test.max() * 1.4
for ax, m in zip(axes, MODELS):
    ax.plot([lo_, hi_], [lo_, hi_], color=INK2, lw=1, ls="--", zorder=1)
    ax.fill_between([lo_, hi_], [lo_ * 0.8, hi_ * 0.8], [lo_ * 1.2, hi_ * 1.2], color=GRID, alpha=0.6, lw=0, zorder=0)
    ax.scatter(y_test, P[m]["Test"], s=9, color=COLORS[m], alpha=0.7, lw=0, zorder=2)
    mm = metrics_full(y_test, P[m]["Test"])
    ax.set_title(f"{m}\nR² {mm['R2']:.3f} · MAPE {mm['MAPE']:.2f}%", loc="left", fontsize=10)
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(lo_, hi_); ax.set_ylim(lo_, hi_)
    ax.set_xlabel("Đơn giá thực tế (đ/m², log)")
axes[0].set_ylabel("Đơn giá dự đoán (đ/m², log)")
fig.suptitle("Thực tế và dự đoán trên tập test — đường chéo: dự đoán đúng; dải xám: ±20%", x=0.01, ha="left", color=INK)
fig.tight_layout(); fig.savefig(HERE / "bieu_do" / "bd2_thuc_te_du_doan.png", dpi=200); plt.close(fig)

# 3. SHAP theo nhóm
s = shap_df.sort_values("mean |SHAP| (log)")
fig, ax = plt.subplots(figsize=(8, 5))
ax.barh(s["nhóm biến"], s["mean |SHAP| (log)"], color=COLORS[REF], height=0.6)
for i, (v, p) in enumerate(zip(s["mean |SHAP| (log)"], s["tỷ trọng %"])):
    ax.text(v + s["mean |SHAP| (log)"].max() * 0.01, i, f"{v:.3f}  ({p:.1f}%)", va="center", fontsize=9, color=INK)
ax.set_xlabel("Trung bình |SHAP| trên tập test (thang log1p đơn giá)")
ax.set_title("LightGBM A1 — đóng góp theo nhóm biến gốc (TreeSHAP)", loc="left")
ax.grid(axis="x", color=GRID, lw=0.8); ax.set_axisbelow(True)
ax.set_xlim(0, s["mean |SHAP| (log)"].max() * 1.3)
fig.tight_layout(); fig.savefig(HERE / "bieu_do" / "bd3_shap_nhom_bien.png", dpi=200); plt.close(fig)

# 4. phân bố sai số
fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 4.6))
xs = np.linspace(0, 50, 501)
for m in MODELS:
    a = np.sort(APE[m]); a1.plot(xs, np.searchsorted(a, xs, side="right") / len(a) * 100,
                                 color=COLORS[m], lw=2, label=m)
for x in (10, 20):
    a1.axvline(x, color=GRID, lw=1, zorder=0)
a1.set_xlabel("Ngưỡng sai số tuyệt đối APE (%)"); a1.set_ylabel("Tỷ lệ dòng test có APE ≤ ngưỡng (%)")
a1.set_title("Phân bố tích luỹ APE, 5 mô hình", loc="left"); a1.legend(frameon=False, loc="lower right")
a1.set_xlim(0, 50); a1.set_ylim(0, 100); a1.grid(color=GRID, lw=0.6); a1.set_axisbelow(True)
pe = (pL - y_test) / y_test * 100
data = [pe[(q == c)] for c in q.categories]
bp = a2.boxplot(data, widths=0.5, patch_artist=True, showfliers=True,
                flierprops=dict(marker="o", ms=3, mfc=COLORS[REF], mec="none", alpha=0.5),
                medianprops=dict(color=INK, lw=1.5), whiskerprops=dict(color=INK2), capprops=dict(color=INK2))
for p_ in bp["boxes"]:
    p_.set(facecolor="#cfe0f6", edgecolor=COLORS[REF])
a2.set_xticks(range(1, 5)); a2.set_xticklabels([str(c) for c in q.categories])
a2.axhline(0, color=INK2, lw=1, ls="--")
a2.set_ylabel("Sai số có dấu (dự đoán − thực)/thực (%)"); a2.set_xlabel("Nhóm phân vị đơn giá thực tế (test)")
a2.set_title("LightGBM A1 — sai số có dấu theo phân vị đơn giá", loc="left")
a2.grid(axis="y", color=GRID, lw=0.6); a2.set_axisbelow(True)
lim = np.percentile(np.abs(pe), 99.5); a2.set_ylim(-lim * 1.1, lim * 1.1)
fig.tight_layout(); fig.savefig(HERE / "bieu_do" / "bd4_phan_bo_sai_so.png", dpi=200); plt.close(fig)

# ---------------- tóm tắt ra màn hình ----------------
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
fmt = {"R2": 4, "MAE": 0, "RMSE": 0, "MAPE": 2, "MdAPE": 2, "MPE (dự đoán−thực, %)": 2,
       "ME (dự đoán−thực, đ/m²)": 0, "APE<10%": 1, "APE<15%": 1, "APE<20%": 1}
print(met.round(fmt).to_string(index=False)); print()
print(sheets["5_test_ma_ts"].round(fmt).to_string(index=False)); print()
print(boot.round(3).to_string(index=False)); print()
print(oot.round(fmt).to_string(index=False)); print(); print(oot_note.to_string(index=False)); print()
print(sheets["8_bo_bien"].round(fmt).to_string(index=False)); print()
print(shap_df.round(4).to_string(index=False)); print("additivity", add_err); print()
print(sheets["10_sai_so_LGB"].round(fmt).to_string(index=False)); print()
print(pd.DataFrame(hp).to_string(index=False)); print(); print(srch.to_string(index=False)); print()
print(sheets["3_hedonic_ghi_chu"].to_string(index=False))
print("in_tr:", in_tr.sum(), "| test Quý_Năm:", test_raw["Quý_Năm"].unique())
