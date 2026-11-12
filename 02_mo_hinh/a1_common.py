"""Hàm dựng đặc trưng của A1 — CHÉP NGUYÊN VĂN từ A1/pipeline_HCM_A1.ipynb (ô 2, 4, 6, 8, 10).

Không đổi logic. Khác duy nhất: ROOT được tìm từ vị trí file này thay vì Path.cwd(),
và các đoạn in/demo/assert minh hoạ của notebook bị bỏ.
Phần cuối (sau dòng "PHẦN BỔ SUNG") là hàm tiện ích mới của so_sanh/, không thuộc A1.
"""
import os, re, unicodedata, warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_absolute_percentage_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from scipy.sparse import hstack

warnings.filterwarnings("ignore")

# ---------------- ô 2: cấu hình ----------------
ROOT = next(p for p in [Path(__file__).resolve().parent, *Path(__file__).resolve().parents]
            if (p / "data" / "processed" / "train_HCM_3_clean.xlsx").exists())
TRAIN_XLS = ROOT / "data" / "processed" / "train_HCM_3_clean.xlsx"
TEST_XLS  = ROOT / "data" / "processed" / "tstd_HCM_3_clean.xlsx"

SEED, TEST_SIZE = 42, 0.2
THREADS = os.cpu_count()                # A đặt 20 theo máy công ty; A1 dùng số luồng của máy đang chạy
CHITIET_SVD_DIM = 24

LGB_PARAMS = dict(objective="regression", learning_rate=0.03, n_estimators=12000,
                  num_leaves=255, min_child_samples=5, feature_fraction=0.8,
                  bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0,
                  num_threads=THREADS, random_state=SEED, verbose=-1)
EARLY_STOP = 200

# ---------------- ô 4: khai báo cột ----------------
TARGET = "Đơn giá quyền sử dụng đất (đ/m2)1"
XA     = "Xã/Phường/Thị trấn mới"
VUNG   = "Vùng"                              # hcm1 / hcm2 (dữ liệu sạch đã viết thường)

CAT_BASE = [XA, VUNG, "Đường phố", "Vị trí trong khung giá", "Quý_Năm",
            "Lợi thế kinh doanh", "hinh_dang_sach"]
NUM_BASE = ["Diện tích (m2)", "Kích thước mặt tiền (m)", "Kích thước chiều dài",
            "Số mặt tiền tiếp giáp", "Độ rộng ngõ/ngách nhỏ nhất (Từ đường chính đến BĐS)",
            "Khoảng cách đến đường chính (m)"]
TEXT_TFIDF = ["Yếu tố khác"]
CHITIET    = "Chi tiết"

BASE_ORDER = [XA, VUNG, "Đường phố",
              "Diện tích (m2)", "Kích thước mặt tiền (m)", "Kích thước chiều dài",
              "Số mặt tiền tiếp giáp", "Độ rộng ngõ/ngách nhỏ nhất (Từ đường chính đến BĐS)",
              "Khoảng cách đến đường chính (m)", "Vị trí trong khung giá",
              "Lợi thế kinh doanh", "hinh_dang_sach", "Quý_Năm"]

EXCLUDED = {
    "tính từ đơn giá":       ["Don_gia_TB", "Min", "Max", "Độ lệch chuẩn", "Số lượng mẫu"],
    "chỉ số":                ["Chi_so_ngo", "Chi_so_dientich", "Chi_so_hinhdang"],
    "cờ True/False":         ["la_hem", "co_mat_tien", "chi_la_mat_tien", "la_so_nha",
                              "co_bang_chung_mat_tien", "da_sua_vt_theo_mat_tien",
                              "kc_thieu_du_lieu", "da_sua_kich_thuoc", "canh_bao_vt1",
                              "canh_bao_mt_vt234"],
    "thay bằng hinh_dang_sach": ["Hình dáng"],
    "tỷ lệ (chỉ để quan sát)":  ["ty_le_kt_dt", "ty_le_dai_rong"],
    "A cũng không dùng":     ["Thành phố/Quận/Huyện/Thị xã", "Ghep", "Đoạn đường", "Mã tài sản", "Tháng"],
}
EXCLUDED_ALL = [c for v in EXCLUDED.values() for c in v]
USED_RAW = CAT_BASE + NUM_BASE + TEXT_TFIDF + [CHITIET]

# ---------------- ô 6: bộ mã hoá Chi tiết ----------------
_URL_RE = re.compile(r"https?://\S+|www\.\S+")
_NUM_RE = re.compile(r"(\d+)\s*$")
_JUNK   = {"", ".", "..", "-", "/", "x", "na", "n a", "nan", "none", "0", "không", "khong"}
_ABBR   = {"kp": "khu phố", "kdc": "khu dân cư", "mt": "mặt tiền", "hxh": "hẻm xe hơi",
           "tđc": "tái định cư", "tdc": "tái định cư", "kcn": "khu công nghiệp",
           "qh": "quy hoạch", "kdt": "khu đô thị", "kđt": "khu đô thị", "cc": "chung cư",
           "hcc": "hẻm cụt", "đ": "đường", "p": "phường", "q": "quận"}

def norm_chitiet(s):
    if pd.isna(s):
        return ""
    s = unicodedata.normalize("NFC", str(s)).replace("​", " ").strip().lower()
    if _URL_RE.search(s):
        return "<url>"
    s = re.sub(r"(?<=[a-zà-ỹ])(?=\d)", " ", s)
    s = re.sub(r"[^\w\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    if s in _JUNK or len(s) <= 1:
        return ""
    return " ".join(_ABBR.get(t, t) for t in s.split()).strip()

def chitiet_flags(norm):
    f = pd.DataFrame(index=norm.index)
    f["ct_len"]      = norm.str.len()
    f["ct_missing"]  = (norm == "").astype(int)
    f["ct_hem"]      = norm.str.contains(r"hẻm|ngõ|ngách|hem", regex=True).astype(int)
    f["ct_mat_tien"] = norm.str.contains(r"mặt tiền|mat tien", regex=True).astype(int)
    f["ct_noi_bo"]   = norm.str.contains(r"nội bộ|noi bo", regex=True).astype(int)
    f["ct_nha"]      = norm.str.contains(r"\bnhà\b|biệt thự|chung cư", regex=True).astype(int)
    f["ct_dat"]      = norm.str.contains(r"\bđất\b", regex=True).astype(int)
    f["ct_du_an"]    = norm.str.contains(r"dự án|khu dân cư|khu đô thị|resort|villa", regex=True).astype(int)
    f["ct_num"]      = norm.str.extract(_NUM_RE)[0].astype(float).fillna(-1)
    return f

CT_FLAGS = ["ct_len", "ct_missing", "ct_hem", "ct_mat_tien", "ct_noi_bo",
            "ct_nha", "ct_dat", "ct_du_an", "ct_num"]

class ChiTietEncoder:
    """Chuẩn hoá + 9 cờ + TF-IDF (từ 1-2, ký tự char_wb 2-4) -> TruncatedSVD."""

    def __init__(self, n_components=CHITIET_SVD_DIM):
        self.n_components = n_components

    def fit(self, s):
        norm = s.map(norm_chitiet)
        self.vw = TfidfVectorizer(analyzer="word",    ngram_range=(1, 2), min_df=3, max_features=600)
        self.vc = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), min_df=5, max_features=1500)
        X = hstack([self.vw.fit_transform(norm), self.vc.fit_transform(norm)]).tocsr()
        self.dim = min(self.n_components, X.shape[1] - 1)
        self.svd = TruncatedSVD(n_components=self.dim, random_state=SEED).fit(X)
        return self

    def transform(self, s):
        norm = s.map(norm_chitiet)
        out = chitiet_flags(norm)
        E = self.svd.transform(hstack([self.vw.transform(norm), self.vc.transform(norm)]).tocsr())
        for i in range(self.dim):
            out[f"ct_emb_{i}"] = E[:, i]
        return out

    @property
    def columns(self):
        return CT_FLAGS + [f"ct_emb_{i}" for i in range(self.dim)]

    def get_state(self):
        return {"vw": self.vw, "vc": self.vc, "svd": self.svd,
                "dim": self.dim, "n_components": self.n_components}

    @classmethod
    def from_state(cls, s):
        o = cls(n_components=s["n_components"])
        o.vw, o.vc, o.svd, o.dim = s["vw"], s["vc"], s["svd"], s["dim"]
        return o

# ---------------- ô 8: nạp dữ liệu ----------------
def load_clean(path, need_target=True):
    df = pd.read_excel(path)
    if need_target:
        df = df[df[TARGET].notna() & (df[TARGET] > 0)].reset_index(drop=True)
    return df

# ---------------- ô 10: dựng đặc trưng ----------------
def clean_base(df, medians=None):
    out = pd.DataFrame(index=df.index)
    for c in CAT_BASE:
        out[c] = df[c].astype(object).where(df[c].notna(), "Unknown").astype(str)
    for c in NUM_BASE:
        out[c] = pd.to_numeric(df[c], errors="coerce")
    if medians is None:
        medians = out[NUM_BASE].median()
    out[NUM_BASE] = out[NUM_BASE].fillna(medians)
    return out, medians

def fit_tfidf(df):
    return {c: TfidfVectorizer(max_features=600, stop_words="english",
                               ngram_range=(1, 2)).fit(df[c].fillna("").astype(str))
            for c in TEXT_TFIDF}

def apply_tfidf(vecs, df):
    blocks = []
    for col, v in vecs.items():
        A_ = v.transform(df[col].fillna("").astype(str)).toarray()
        pref = f'tfidf_{col.replace(" ", "_").lower()}_'
        blocks.append(pd.DataFrame(A_, columns=[f"{pref}{i}" for i in range(A_.shape[1])], index=df.index))
    return pd.concat(blocks, axis=1)

def assemble(base, ct_df, tfidf_df):
    X = pd.concat([base.reset_index(drop=True), ct_df.reset_index(drop=True),
                   tfidf_df.reset_index(drop=True)], axis=1)
    return X[[c for c in BASE_ORDER if c in X.columns] + [c for c in X.columns if c not in BASE_ORDER]]

def metrics(y_true, y_pred, name):
    a = np.abs((y_true - y_pred) / (y_true + 1e-10)) * 100
    return {"tập": name, "n": len(y_true), "R2": r2_score(y_true, y_pred),
            "MAPE": mean_absolute_percentage_error(y_true, y_pred) * 100,
            "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
            "APE<10%": float((a < 10).mean() * 100), "APE<15%": float((a < 15).mean() * 100)}

# ô 12
def to_cat(X, CATEGORIES):           # A1 dùng biến toàn cục CATEGORIES; ở đây truyền vào tham số
    return X.assign(**{c: pd.Categorical(X[c], categories=CATEGORIES[c]) for c in CAT_BASE})


# ======================= PHẦN BỔ SUNG (không thuộc A1) =======================
def build_features(tr_raw, va_raw, te_raw):
    """Lặp lại đúng trình tự ô 10 của A1: fit medians/Chi tiết/TF-IDF trên tr_raw rồi áp cho va/te."""
    b_tr, medians = clean_base(tr_raw)
    b_va, _ = clean_base(va_raw, medians); b_te, _ = clean_base(te_raw, medians)
    ct_enc = ChiTietEncoder().fit(tr_raw[CHITIET])
    yt_vecs = fit_tfidf(tr_raw)
    X_tr = assemble(b_tr, ct_enc.transform(tr_raw[CHITIET]), apply_tfidf(yt_vecs, tr_raw))
    X_va = assemble(b_va, ct_enc.transform(va_raw[CHITIET]), apply_tfidf(yt_vecs, va_raw))[X_tr.columns]
    X_te = assemble(b_te, ct_enc.transform(te_raw[CHITIET]), apply_tfidf(yt_vecs, te_raw))[X_tr.columns]
    CATEGORIES = {c: list(pd.unique(X_tr[c])) for c in CAT_BASE}
    return X_tr, X_va, X_te, CATEGORIES


def metrics_full(y_true, y_pred, name=None):
    """Chỉ số mở rộng cho so_sanh/: R2, MAE, RMSE, MAPE, MdAPE, sai số có dấu, APE<10/15/20%."""
    y_true = np.asarray(y_true, float); y_pred = np.asarray(y_pred, float)
    e = y_pred - y_true
    pe = e / y_true * 100
    a = np.abs(pe)
    d = {"n": len(y_true),
         "R2": r2_score(y_true, y_pred) if len(y_true) > 1 else np.nan,
         "MAE": np.mean(np.abs(e)), "RMSE": np.sqrt(np.mean(e ** 2)),
         "MAPE": a.mean(), "MdAPE": np.median(a),
         "MPE (dự đoán−thực, %)": pe.mean(), "ME (dự đoán−thực, đ/m²)": e.mean(),
         "APE<10%": (a < 10).mean() * 100, "APE<15%": (a < 15).mean() * 100,
         "APE<20%": (a < 20).mean() * 100}
    return ({"tập": name, **d} if name is not None else d)
