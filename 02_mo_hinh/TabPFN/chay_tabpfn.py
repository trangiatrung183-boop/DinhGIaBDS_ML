"""Chạy thử TabPFN (v2, GPU) trên cùng dữ liệu / cách chia / chỉ số với các mô hình ở 02_mo_hinh/.

    python 02_mo_hinh/TabPFN/chay_tabpfn.py pilot     # 2.000 dòng ngữ cảnh, 1 tập con: đo thời gian / VRAM / RAM
    python 02_mo_hinh/TabPFN/chay_tabpfn.py main      # chạy chính + ngoài thời gian (có checkpoint)
    python 02_mo_hinh/TabPFN/chay_tabpfn.py report    # chỉ đọc checkpoint, xuất Excel / CSV / PNG

Khác với LightGBM:
- TabPFN v2 không huấn luyện; nó dùng tối đa ~10.000 dòng làm "ngữ cảnh". Train có 34.772 dòng
  nên lấy N_SUB tập con ngẫu nhiên, mỗi tập N_CTX dòng, dự đoán trên thang log1p rồi lấy trung bình.
- Chỉ dùng các cột gốc + 33 cột "Chi tiết" của A1 (không dùng 600 cột TF-IDF).
- Phường/xã (173 mức) và Đường phố (2.701 mức) mã hoá bằng target encoding có làm mượt
  (train: out-of-fold 5 phần; val/test: ánh xạ học trên toàn bộ train) + số lần xuất hiện.
- Không chọn siêu tham số theo test; cấu hình cố định, chỉ val dùng để so sánh số tập con.
- v2.5+ cần đăng nhập để chấp nhận giấy phép nên không dùng ở đây.
"""
import sys, os, time, json, re
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import numpy as np
import pandas as pd
import joblib

HERE = Path(__file__).resolve().parent
SS = HERE.parent
sys.path.insert(0, str(SS))
from a1_common import *          # noqa: F401,F403
from sklearn.model_selection import KFold

CK = HERE / "checkpoints"; CK.mkdir(exist_ok=True)
CK_SS = SS / "checkpoints"
QUAN = "Thành phố/Quận/Huyện/Thị xã"
OOT_Q = "q2-2026"
N_BOOT = 5000

CFG = dict(N_CTX=int(os.environ.get("N_CTX", 10000)), N_SUB=int(os.environ.get("N_SUB", 3)),
           N_EST=int(os.environ.get("N_EST", 8)), BATCH=int(os.environ.get("BATCH", 512)), VAL_SUB=int(os.environ.get("VAL_SUB", 2000)),
           TE_SMOOTH=20, TE_FOLDS=5, model="TabPFN v2 (Prior-Labs/TabPFN-v2-reg)")


def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(HERE / "log.txt", "a", encoding="utf-8") as f:
        f.write(line + "\n")


def ckpt(name, fn):
    p = CK / f"{name}.joblib"
    if p.exists():
        return joblib.load(p)
    out = fn()
    joblib.dump(out, p, compress=3)
    return out


# ============================ DỮ LIỆU (giống A1) ============================
train_raw = load_clean(TRAIN_XLS); test_raw = load_clean(TEST_XLS)
y_all = train_raw[TARGET].astype(float).values; y_test = test_raw[TARGET].astype(float).values
tr_idx, va_idx = train_test_split(np.arange(len(train_raw)), test_size=TEST_SIZE, random_state=SEED)
tr_raw, va_raw = train_raw.iloc[tr_idx], train_raw.iloc[va_idx]
y_tr, y_va = y_all[tr_idx], y_all[va_idx]

TE_COLS = [XA, "Đường phố"]
ORD_COLS = [VUNG, "Vị trí trong khung giá", "Lợi thế kinh doanh", "hinh_dang_sach"]


def quy_so(s):
    m = s.astype(str).str.extract(r"q(\d)-(\d{4})")
    return m[1].astype(float) + (m[0].astype(float) - 1) / 4


def make_tab(Xtr, ytr, others):
    """Từ bảng đặc trưng A1 -> ma trận số cho TabPFN. Mọi thống kê học trên Xtr."""
    ly = np.log1p(ytr); g = ly.mean(); k = CFG["TE_SMOOTH"]
    keep_num = NUM_BASE + [c for c in Xtr.columns if c.startswith("ct_")]

    def te_map(keys, vals):
        d = pd.DataFrame({"k": keys, "v": vals}).groupby("k")["v"].agg(["sum", "count"])
        return ((d["sum"] + k * g) / (d["count"] + k)), d["count"]

    def base(X):
        B = X[keep_num].astype(float).reset_index(drop=True).copy()
        B["quy_nam_so"] = quy_so(X["Quý_Năm"]).values
        return B

    out_tr = base(Xtr)
    for c in TE_COLS:                                      # out-of-fold cho train
        oof = np.full(len(Xtr), g)
        for a, b in KFold(CFG["TE_FOLDS"], shuffle=True, random_state=SEED).split(Xtr):
            mp, _ = te_map(Xtr[c].values[a], ly[a])
            oof[b] = pd.Series(Xtr[c].values[b]).map(mp).fillna(g).values
        _, cnt = te_map(Xtr[c].values, ly)
        out_tr[f"te_{c}"] = oof
        out_tr[f"dem_{c}"] = pd.Series(Xtr[c].values).map(cnt).fillna(0).values
    levels = {c: {v: i for i, v in enumerate(sorted(Xtr[c].unique()))} for c in ORD_COLS}
    for c in ORD_COLS:
        out_tr[f"ma_{c}"] = pd.Series(Xtr[c].values).map(levels[c]).fillna(-1).values

    res = [out_tr]
    for X in others:
        o = base(X)
        for c in TE_COLS:
            mp, cnt = te_map(Xtr[c].values, ly)
            o[f"te_{c}"] = pd.Series(X[c].values).map(mp).fillna(g).values
            o[f"dem_{c}"] = pd.Series(X[c].values).map(cnt).fillna(0).values
        for c in ORD_COLS:
            o[f"ma_{c}"] = pd.Series(X[c].values).map(levels[c]).fillna(-1).values
        res.append(o[out_tr.columns])
    cat_idx = [list(out_tr.columns).index(f"ma_{c}") for c in ORD_COLS]
    return res, cat_idx


# ============================ TABPFN ============================
def run_tabpfn(Xtr, ytr, X_pred_list, n_ctx, n_sub, tag):
    """Dự đoán trung bình (thang log1p) qua n_sub tập con ngữ cảnh; checkpoint từng tập con."""
    import torch, psutil
    from tabpfn import TabPFNRegressor
    from tabpfn.constants import ModelVersion
    (Ttr, *Tp), cat_idx = make_tab(Xtr, ytr, X_pred_list)
    ly = np.log1p(ytr)
    rng = np.random.default_rng(SEED)
    subs = [rng.choice(len(Ttr), size=min(n_ctx, len(Ttr)), replace=False) for _ in range(n_sub)]
    preds, res = [], []
    for s, idx in enumerate(subs):
        def _one(idx=idx, s=s):
            torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
            m = TabPFNRegressor.create_default_for_version(
                ModelVersion.V2, device="cuda", n_estimators=CFG["N_EST"], random_state=SEED + s,
                categorical_features_indices=cat_idx, ignore_pretraining_limits=True,
                fit_mode="fit_preprocessors")
            t0 = time.time()
            m.fit(Ttr.values[idx], ly[idx])
            out = []
            for T in Tp:
                out.append(np.concatenate([m.predict(T.values[i:i + CFG["BATCH"]])
                                           for i in range(0, len(T), CFG["BATCH"])]))
            torch.cuda.synchronize()
            r = {"pred_log": out, "secs": round(time.time() - t0, 1),
                 "vram_peak_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2),
                 "ram_gb": round(psutil.Process().memory_info().rss / 1e9, 2),
                 "n_ctx": len(idx), "n_pred": int(sum(len(T) for T in Tp))}
            del m; torch.cuda.empty_cache()
            return r
        r = ckpt(f"{tag}_sub{s}", _one)
        log(f"{tag} tập con {s + 1}/{n_sub}: ngữ cảnh {r['n_ctx']} dòng, dự đoán {r['n_pred']} dòng, "
            f"{r['secs']}s, VRAM đỉnh {r['vram_peak_gb']} GB, RAM tiến trình {r['ram_gb']} GB")
        preds.append(r["pred_log"]); res.append({k: v for k, v in r.items() if k != "pred_log"})
    avg = [np.expm1(np.mean([p[j] for p in preds], axis=0)) for j in range(len(X_pred_list))]
    each = [[np.expm1(p[j]) for p in preds] for j in range(len(X_pred_list))]
    return {"pred": avg, "pred_each": each, "res": res, "n_feat": Ttr.shape[1], "features": list(Ttr.columns)}


def mape(y, p):
    return float(np.mean(np.abs((p - y) / y)) * 100)


X_tr, X_va, X_te, _ = joblib.load(CK_SS / "features_main.joblib")
mode = sys.argv[1] if len(sys.argv) > 1 else "main"

if mode == "pilot":
    import torch
    log(f"PILOT · torch {torch.__version__} · GPU {torch.cuda.get_device_name(0)} · cấu hình {json.dumps(CFG, ensure_ascii=False)}")
    r = ckpt("pilot", lambda: run_tabpfn(X_tr, y_tr, [X_va, X_te], 2000, 1, "pilot_raw"))
    log(f"PILOT xong: {r['n_feat']} đặc trưng | Val MAPE {mape(y_va, r['pred'][0]):.2f}% · Test MAPE {mape(y_test, r['pred'][1]):.2f}%")
    sys.exit(0)

if mode == "main":
    log(f"MAIN · cấu hình {json.dumps(CFG, ensure_ascii=False)}")
    # val chỉ dùng mẫu VAL_SUB dòng (cố định theo SEED) để giới hạn thời gian; test dùng đủ 693 dòng
    va_sel = np.sort(np.random.default_rng(SEED).choice(len(X_va), size=min(CFG["VAL_SUB"], len(X_va)), replace=False))
    r = run_tabpfn(X_tr, y_tr, [X_va.iloc[va_sel], X_te], CFG["N_CTX"], CFG["N_SUB"], f"main_{CFG['N_CTX']}")
    r["va_sel"] = va_sel; r["cfg"] = CFG
    joblib.dump(r, CK / "main.joblib", compress=3)
    log(f"TabPFN chính: Val(mẫu {len(va_sel)}) MAPE {mape(y_va[va_sel], r['pred'][0]):.2f}% · Test MAPE {mape(y_test, r['pred'][1]):.2f}%")

    oot_mask = (train_raw["Quý_Năm"].astype(str) != OOT_Q).values
    o_raw = train_raw[oot_mask].reset_index(drop=True); y_o = y_all[oot_mask]
    o_tr_idx, _ = train_test_split(np.arange(len(o_raw)), test_size=TEST_SIZE, random_state=SEED)
    O_tr, _, O_te, _ = joblib.load(CK_SS / "features_oot.joblib")
    ro = run_tabpfn(O_tr, y_o[o_tr_idx], [O_te], CFG["N_CTX"], CFG["N_SUB"], f"oot_{CFG['N_CTX']}")
    joblib.dump(ro, CK / "oot.joblib", compress=3)
    log(f"TabPFN ngoài thời gian: tstd MAPE {mape(y_test, ro['pred'][0]):.2f}%")
    ro["cfg"] = CFG; joblib.dump(ro, CK / "oot.joblib", compress=3)
    log("XONG main")
