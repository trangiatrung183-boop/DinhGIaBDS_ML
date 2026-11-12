"""Bước 2, 3, 7, 8: huấn luyện mọi mô hình, có checkpoint (chạy lại sẽ bỏ qua phần đã xong).

    python 02_mo_hinh/chay_mo_hinh.py

Checkpoint: 02_mo_hinh/checkpoints/*.joblib (dự đoán + tham số + nhật ký), mô hình tốt nhất lưu
ở 02_mo_hinh/checkpoints/models/. Xoá file checkpoint tương ứng để chạy lại một phần.
"""
import sys, time, json
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
from scipy import stats, linalg

sys.path.insert(0, str(Path(__file__).resolve().parent))
from a1_common import *          # noqa: F401,F403  (hàm A1 chép nguyên văn)
import a1_common as A

HERE = Path(__file__).resolve().parent
CK = HERE / "checkpoints"; (CK / "models").mkdir(parents=True, exist_ok=True)
N_SEARCH = 6                     # số cấu hình random search mỗi mô hình (<= 8)
OOT_Q = "q2-2026"


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


def mape(y, p):
    return float(np.mean(np.abs((p - y) / y)) * 100)


# ============================ DỮ LIỆU ============================
train_raw = load_clean(TRAIN_XLS)
test_raw = load_clean(TEST_XLS)
y_train_all = train_raw[TARGET].astype(float).values
y_test = test_raw[TARGET].astype(float).values
tr_idx, va_idx = train_test_split(np.arange(len(train_raw)), test_size=TEST_SIZE, random_state=SEED)
tr_raw, va_raw = train_raw.iloc[tr_idx], train_raw.iloc[va_idx]
y_tr, y_va = y_train_all[tr_idx], y_train_all[va_idx]

def _feats():
    return build_features(tr_raw, va_raw, test_raw)
X_tr, X_va, X_te, CATS = ckpt("features_main", _feats)
FEATURES = list(X_tr.columns)
log(f"Đặc trưng: {len(FEATURES)} | train {len(X_tr)} · val {len(X_va)} · test {len(X_te)}")


# ============================ LIGHTGBM ============================
def train_lgb(Xtr, ytr, Xva, yva, Xte_list, cats, feats=None):
    feats = feats or list(Xtr.columns)
    catcols = [c for c in CAT_BASE if c in feats]
    tc = lambda X: X[feats].assign(**{c: pd.Categorical(X[c], categories=cats[c]) for c in catcols})
    t0 = time.time()
    m = lgb.LGBMRegressor(**LGB_PARAMS)
    m.fit(tc(Xtr), np.log1p(ytr), eval_set=[(tc(Xva), np.log1p(yva))],
          categorical_feature=catcols,
          callbacks=[lgb.early_stopping(EARLY_STOP, verbose=False)])
    preds = [np.expm1(m.predict(tc(X))) for X in [Xtr, Xva] + Xte_list]
    return m, preds, round(time.time() - t0, 1)

import lightgbm as lgb

def _lgb_main():
    m, (ptr, pva, pte), secs = train_lgb(X_tr, y_tr, X_va, y_va, [X_te], CATS)
    m.booster_.save_model(str(CK / "models" / "lightgbm_A1.txt"))
    return {"pred": {"Train": ptr, "Val": pva, "Test": pte}, "secs": secs,
            "best_iter": int(m.best_iteration_), "params": {**LGB_PARAMS, "early_stopping": EARLY_STOP}}

r = ckpt("m_lightgbm", _lgb_main)
man = json.load(open(ROOT / "02_mo_hinh" / "LightGBM" / "model" / "manifest.json", encoding="utf-8"))
ref = {d["tập"]: d for d in man["results"]["metrics"]}
mt = metrics(y_test, r["pred"]["Test"], "Test")
dR2, dM = abs(mt["R2"] - ref["Test"]["R2"]), abs(mt["MAPE"] - ref["Test"]["MAPE"])
log(f"LightGBM A1: {r['secs']}s, best_iter {r['best_iter']} (manifest {man['results']['best_iter']}) | "
    f"Test R2 {mt['R2']:.4f} (manifest {ref['Test']['R2']:.4f}) MAPE {mt['MAPE']:.2f}% (manifest {ref['Test']['MAPE']:.2f}%)")
if dR2 > 1e-3 or dM > 0.05:
    log(f"DỪNG: LightGBM lệch manifest (ΔR2={dR2:.4g}, ΔMAPE={dM:.4g}). Không chạy tiếp.")
    sys.exit(1)


# ============================ HEDONIC OLS ============================
HED_CAT = [XA, VUNG, "Vị trí trong khung giá", "Lợi thế kinh doanh", "hinh_dang_sach", "Quý_Năm"]
HED_NUM = {  # biến số, dạng log cho biến quy mô (chuẩn hàm hedonic log-log)
    "ln_dien_tich": ("Diện tích (m2)", np.log),
    "ln_mat_tien": ("Kích thước mặt tiền (m)", np.log),
    "ln_chieu_dai": ("Kích thước chiều dài", np.log),
    "so_mat_tien": ("Số mặt tiền tiếp giáp", lambda v: v),
    "ln1p_do_rong_ngo": ("Độ rộng ngõ/ngách nhỏ nhất (Từ đường chính đến BĐS)", np.log1p),
    "ln1p_khoang_cach": ("Khoảng cách đến đường chính (m)", np.log1p),
}

def hed_design(X, levels=None):
    """X: bảng đã qua clean_base (cột phân loại là chuỗi, số đã điền trung vị train)."""
    D = pd.DataFrame(index=X.index)
    for k, (c, f) in HED_NUM.items():
        v = X[c].astype(float).clip(lower=1e-3) if f is np.log else X[c].astype(float).clip(lower=0)
        D[k] = f(v)
    if levels is None:   # mức tham chiếu = mức phổ biến nhất trong train
        levels = {c: [l for l in X[c].value_counts().index[1:]] for c in HED_CAT}
    for c in HED_CAT:
        for l in levels[c]:
            D[f"{c}={l}"] = (X[c] == l).astype(float)
    return D, levels

def _hedonic():
    t0 = time.time()
    Dtr, lv = hed_design(X_tr)
    # bỏ cột phụ thuộc tuyến tính (QR có pivot) để hệ số xác định duy nhất
    Z = np.column_stack([np.ones(len(Dtr)), Dtr.values])
    names = ["const"] + list(Dtr.columns)
    Q, R, piv = linalg.qr(Z, mode="economic", pivoting=True)
    tol = np.abs(R[0, 0]) * max(Z.shape) * np.finfo(float).eps
    rank = int((np.abs(np.diag(R)) > tol).sum())
    keep = sorted(piv[:rank]); dropped = [names[i] for i in sorted(piv[rank:])]
    Z = Z[:, keep]; names = [names[i] for i in keep]
    yl = np.log1p(y_tr)
    XtX_inv = np.linalg.inv(Z.T @ Z)
    beta = XtX_inv @ Z.T @ yl
    res = yl - Z @ beta
    n, k = Z.shape
    s2 = res @ res / (n - k)
    se = np.sqrt(np.diag(XtX_inv) * s2)
    # sai số chuẩn vững HC1
    meat = (Z * res[:, None] ** 2).T @ Z
    se_hc1 = np.sqrt(np.diag(XtX_inv @ meat @ XtX_inv) * n / (n - k))
    t, t_hc = beta / se, beta / se_hc1
    coef = pd.DataFrame({"biến": names, "hệ số": beta, "SE": se, "t": t,
                         "p-value": 2 * stats.t.sf(np.abs(t), n - k),
                         "SE (HC1)": se_hc1, "p-value (HC1)": 2 * stats.t.sf(np.abs(t_hc), n - k),
                         "% thay đổi đơn giá (exp(b)-1)": (np.exp(beta) - 1) * 100})
    r2_log = 1 - res @ res / ((yl - yl.mean()) @ (yl - yl.mean()))
    ref_lv = {c: X_tr[c].value_counts().index[0] for c in HED_CAT}

    def pred(X):
        D, _ = hed_design(X, lv)
        Zx = np.column_stack([np.ones(len(D)), D.values])[:, keep]
        return np.expm1(Zx @ beta)
    unseen = {c: int((~X_te[c].isin(set(X_tr[c]))).sum()) for c in HED_CAT}
    return {"pred": {"Train": pred(X_tr), "Val": pred(X_va), "Test": pred(X_te)},
            "coef": coef, "dropped": dropped, "ref_levels": ref_lv, "unseen_test": unseen,
            "n": n, "k": k, "R2_log_train": r2_log, "secs": round(time.time() - t0, 1),
            "params": {"spec": "OLS log1p(đơn giá) ~ " + ", ".join(HED_NUM) + " + one-hot " + ", ".join(HED_CAT)},
            "beta": beta, "keep": keep, "levels": lv}

h = ckpt("m_hedonic", _hedonic)
log(f"Hedonic OLS: k={h['k']} biến, bỏ {len(h['dropped'])} cột cộng tuyến, R2(log, train)={h['R2_log_train']:.3f} | "
    f"Val MAPE {mape(y_va, h['pred']['Val']):.2f}% · Test MAPE {mape(y_test, h['pred']['Test']):.2f}%")


# ======================= RANDOM SEARCH CHUNG =======================
def sample_configs(grid, n, seed):
    rng = np.random.default_rng(seed)
    seen, out = set(), []
    while len(out) < n:
        c = {k: v[rng.integers(len(v))] for k, v in grid.items()}
        key = json.dumps(c, default=str, sort_keys=True)
        if key not in seen:
            seen.add(key); out.append(c)
    return out

def search(name, grid, fit_fn, seed):
    """fit_fn(cfg) -> dict(pred={'Train','Val','Test'}, ...). Chọn cấu hình theo MAPE Val (không nhìn Test)."""
    cfgs = sample_configs(grid, N_SEARCH, seed)
    rows, best = [], None
    for i, cfg in enumerate(cfgs):
        r = ckpt(f"search_{name}_{i}", lambda cfg=cfg: fit_fn(cfg))
        vm = mape(y_va, r["pred"]["Val"])
        rows.append({"mô hình": name, "cấu hình": i, **{k: str(v) for k, v in cfg.items()},
                     "best_iter": r.get("best_iter"), "giây": r["secs"], "MAPE Val": vm})
        if best is None or vm < best[0]:
            best = (vm, i, r, cfg)
    vm, i, r, cfg = best
    r = {k: v for k, v in r.items() if k != "model"}
    return {**r, "params": cfg, "search_log": pd.DataFrame(rows), "chosen": i}


# ============================ RANDOM FOREST ============================
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.preprocessing import OrdinalEncoder

def ord_encode(Xtr, others, cols, max_levels=None):
    """Mã hoá thứ tự cho cột phân loại; max_levels: giữ (max_levels-1) mức phổ biến nhất, còn lại gộp '__khac__'."""
    keep = {}
    def red(X):
        X = X.copy()
        for c in cols:
            if max_levels is not None:
                X[c] = X[c].where(X[c].isin(keep[c]), "__khac__")
        return X
    if max_levels is not None:
        keep = {c: set(Xtr[c].value_counts().index[:max_levels - 1]) for c in cols}
    enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=np.nan,
                         encoded_missing_value=np.nan).fit(red(Xtr)[cols])
    def tr(X):
        X = red(X); out = X.copy()
        out[cols] = enc.transform(X[cols])
        return out.astype(float)
    return [tr(Xtr)] + [tr(X) for X in others]

def _rf_data():
    return ord_encode(X_tr, [X_va, X_te], CAT_BASE)
R_tr, R_va, R_te = ckpt("features_ordinal", _rf_data)
# RF sklearn không nhận NaN ở mọi phiên bản cũ; 1.9 nhận, nhưng điền -1 cho chắc chắn & minh bạch
R_tr, R_va, R_te = [X.fillna(-1) for X in (R_tr, R_va, R_te)]

def fit_rf(cfg):
    t0 = time.time()
    m = RandomForestRegressor(n_estimators=300, n_jobs=THREADS, random_state=SEED, **cfg)
    m.fit(R_tr.values, np.log1p(y_tr))
    pred = {k: np.expm1(m.predict(X.values)) for k, X in [("Train", R_tr), ("Val", R_va), ("Test", R_te)]}
    return {"pred": pred, "secs": round(time.time() - t0, 1)}

RF_GRID = {"max_features": [0.2, 0.33, 0.5, "sqrt"], "min_samples_leaf": [1, 2, 5],
           "max_depth": [None, 30]}
rf = ckpt("m_rf", lambda: search("RandomForest", RF_GRID, fit_rf, SEED + 1))
log(f"Random Forest: chọn {rf['params']} theo Val | Val MAPE {mape(y_va, rf['pred']['Val']):.2f}% · "
    f"Test MAPE {mape(y_test, rf['pred']['Test']):.2f}%")


# ============================ HIST GRADIENT BOOSTING ============================
# HGB: biến phân loại native cần mã ≤ max_bins (255) mức -> giữ 254 mức phổ biến nhất mỗi cột, gộp phần còn lại.
def _hgb_data():
    return ord_encode(X_tr, [X_va, X_te], CAT_BASE, max_levels=254)
H_tr, H_va, H_te = ckpt("features_hgb", _hgb_data)
HGB_CAT_MASK = [c in CAT_BASE for c in H_tr.columns]

def fit_hgb(cfg, step=100, patience=3, max_iter=5000):
    t0 = time.time()
    m = HistGradientBoostingRegressor(categorical_features=HGB_CAT_MASK, early_stopping=False,
                                      warm_start=True, random_state=SEED, max_iter=step, **cfg)
    lva, best, bad = np.log1p(y_va), np.inf, 0
    while True:   # dừng sớm trên Val (A1 cũng dừng sớm trên Val), bước 100 vòng, kiên nhẫn 300 vòng
        m.fit(H_tr.values, np.log1p(y_tr))
        loss = np.mean((m.predict(H_va.values) - lva) ** 2)
        if loss < best - 1e-7: best, bad = loss, 0
        else: bad += 1
        if bad >= patience or m.max_iter >= max_iter: break
        m.max_iter += step
    val_curve = [np.mean((p - lva) ** 2) for p in m.staged_predict(H_va.values)]
    bi = int(np.argmin(val_curve))
    def at(X):
        for i, p in enumerate(m.staged_predict(X.values)):
            if i == bi: return np.expm1(p)
    pred = {k: at(X) for k, X in [("Train", H_tr), ("Val", H_va), ("Test", H_te)]}
    return {"pred": pred, "secs": round(time.time() - t0, 1), "best_iter": bi + 1}

HGB_GRID = {"learning_rate": [0.05, 0.1], "max_leaf_nodes": [31, 63, 127, 255],
            "min_samples_leaf": [5, 10, 20], "l2_regularization": [0.0, 1.0],
            "max_features": [0.5, 0.8, 1.0]}
hgb = ckpt("m_hgb", lambda: search("HistGradientBoosting", HGB_GRID, fit_hgb, SEED + 2))
log(f"HistGradientBoosting: chọn {hgb['params']}, best_iter {hgb['best_iter']} | "
    f"Val MAPE {mape(y_va, hgb['pred']['Val']):.2f}% · Test MAPE {mape(y_test, hgb['pred']['Test']):.2f}%")


# ============================ CATBOOST ============================
from catboost import CatBoostRegressor, Pool

def fit_cat(cfg, Xtr=X_tr, ytr=y_tr, Xva=X_va, yva=y_va, Xte=X_te, save=None):
    t0 = time.time()
    m = CatBoostRegressor(iterations=5000, loss_function="RMSE", random_seed=SEED,
                          thread_count=THREADS, od_type="Iter", od_wait=200, use_best_model=True,
                          verbose=False, allow_writing_files=False, **cfg)
    m.fit(Pool(Xtr, np.log1p(ytr), cat_features=CAT_BASE),
          eval_set=Pool(Xva, np.log1p(yva), cat_features=CAT_BASE))
    if save: m.save_model(str(save))
    pred = {k: np.expm1(m.predict(Pool(X, cat_features=CAT_BASE)))
            for k, X in [("Train", Xtr), ("Val", Xva), ("Test", Xte)]}
    return {"pred": pred, "secs": round(time.time() - t0, 1), "best_iter": int(m.get_best_iteration()) + 1}

CAT_GRID = {"depth": [6, 8], "learning_rate": [0.05, 0.1], "l2_leaf_reg": [1, 3, 10],
            "one_hot_max_size": [4, 10]}
cb = ckpt("m_catboost", lambda: search("CatBoost", CAT_GRID, fit_cat, SEED + 3))
log(f"CatBoost: chọn {cb['params']}, best_iter {cb['best_iter']} | "
    f"Val MAPE {mape(y_va, cb['pred']['Val']):.2f}% · Test MAPE {mape(y_test, cb['pred']['Test']):.2f}%")


# ======================= BƯỚC 7: NGOÀI THỜI GIAN =======================
OTHERS = {"Hedonic OLS": h, "Random Forest": rf, "HistGradientBoosting": hgb, "CatBoost": cb}
best_other = min(OTHERS, key=lambda k: mape(y_va, OTHERS[k]["pred"]["Val"]))   # chọn theo VAL
log(f"Mô hình tốt nhất còn lại (theo Val MAPE): {best_other}")

oot_mask = (train_raw["Quý_Năm"].astype(str) != OOT_Q).values
o_raw = train_raw[oot_mask].reset_index(drop=True); y_o = y_train_all[oot_mask]
o_tr_idx, o_va_idx = train_test_split(np.arange(len(o_raw)), test_size=TEST_SIZE, random_state=SEED)
o_tr, o_va = o_raw.iloc[o_tr_idx], o_raw.iloc[o_va_idx]
yo_tr, yo_va = y_o[o_tr_idx], y_o[o_va_idx]
O_tr, O_va, O_te, O_CATS = ckpt("features_oot", lambda: build_features(o_tr, o_va, test_raw))

def _oot_lgb():
    m, (a, b, c), secs = train_lgb(O_tr, yo_tr, O_va, yo_va, [O_te], O_CATS)
    return {"pred": {"Train": a, "Val": b, "Test": c}, "secs": secs, "best_iter": int(m.best_iteration_),
            "n_train": len(O_tr), "n_val": len(O_va)}
oot_l = ckpt("oot_lightgbm", _oot_lgb)
log(f"Ngoài thời gian LightGBM: train {oot_l['n_train']} dòng (≠{OOT_Q}), best_iter {oot_l['best_iter']} | "
    f"tstd MAPE {mape(y_test, oot_l['pred']['Test']):.2f}%")

def _oot_other():
    cfg = OTHERS[best_other]["params"]
    if best_other == "CatBoost":
        r = fit_cat(cfg, O_tr, yo_tr, O_va, yo_va, O_te)
    elif best_other == "Random Forest":
        a, b, c = [X.fillna(-1) for X in ord_encode(O_tr, [O_va, O_te], CAT_BASE)]
        m = RandomForestRegressor(n_estimators=300, n_jobs=THREADS, random_state=SEED, **cfg).fit(a.values, np.log1p(yo_tr))
        r = {"pred": {k: np.expm1(m.predict(X.values)) for k, X in [("Train", a), ("Val", b), ("Test", c)]}}
    elif best_other == "HistGradientBoosting":
        global H_tr, H_va, H_te, y_tr, y_va
        sv = (H_tr, H_va, H_te, y_tr, y_va)
        H_tr, H_va, H_te = ord_encode(O_tr, [O_va, O_te], CAT_BASE, max_levels=254); y_tr, y_va = yo_tr, yo_va
        r = fit_hgb(cfg)
        H_tr, H_va, H_te, y_tr, y_va = sv
    else:
        raise RuntimeError("Hedonic không được dùng cho bước 7 trong script này")
    return {**r, "model_name": best_other, "params": cfg}
oot_o = ckpt("oot_other", _oot_other)
log(f"Ngoài thời gian {oot_o['model_name']}: tstd MAPE {mape(y_test, oot_o['pred']['Test']):.2f}%")


# ======================= BƯỚC 8: BỎ NHÓM BIẾN =======================
CT_COLS = [c for c in FEATURES if c.startswith("ct_")]
TF_COLS = [c for c in FEATURES if c.startswith("tfidf_")]
ABL = {"a_bo_ChiTiet_YeuToKhac": CT_COLS + TF_COLS,
       "b_bo_DuongPho": ["Đường phố"],
       "c_bo_ChiTiet_YeuToKhac_DuongPho": CT_COLS + TF_COLS + ["Đường phố"]}
for key, drop in ABL.items():
    def _abl(drop=drop):
        feats = [c for c in FEATURES if c not in drop]
        m, (a, b, c), secs = train_lgb(X_tr, y_tr, X_va, y_va, [X_te], CATS, feats)
        return {"pred": {"Train": a, "Val": b, "Test": c}, "secs": secs, "best_iter": int(m.best_iteration_),
                "n_features": len(feats), "dropped": len(drop)}
    ra = ckpt(f"abl_{key}", _abl)
    log(f"Bỏ biến {key}: {ra['n_features']} đặc trưng, best_iter {ra['best_iter']} | "
        f"Val MAPE {mape(y_va, ra['pred']['Val']):.2f}% · Test MAPE {mape(y_test, ra['pred']['Test']):.2f}%")

log("XONG chay_mo_hinh.py")
