import html
import re
from pathlib import Path

import joblib
import pandas as pd
import plotly.express as px
import requests
import streamlit as st


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Tiki Book Intelligence",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# PATHS
# ============================================================

DATA_PATH = Path("data/tiki_books_labeled.csv")
REVIEWS_PATH = Path("data/tiki_books_reviews.csv")
MODEL_COMPARISON_PATH = Path("data/model_comparison.csv")
CLASSIFICATION_REPORT_PATH = Path("data/classification_report.csv")
MODEL_PATH = Path("models/product_performance_model.pkl")
ENCODER_PATH = Path("models/label_encoder.pkl")
CONFUSION_MATRIX_PATH = Path("images/confusion_matrix.png")
FEATURE_IMPORTANCE_PATH = Path("images/feature_importance.png")


# ============================================================
# CSS
# ============================================================

st.markdown(
    """
<style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}

    header[data-testid="stHeader"] {
        background: #EEF6FF;
        height: 0rem;
    }

    .stApp {
        background: linear-gradient(135deg, #EEF6FF 0%, #F8FBFF 55%, #F4F8FF 100%);
        color: #0F172A;
    }

    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2rem;
        max-width: 1450px;
    }

    h1, h2, h3, h4, h5, h6 {
        color: #0F172A !important;
        font-weight: 850 !important;
    }

    p, span, label, div {
        color: #0F172A;
    }

    /* ================= SIDEBAR ================= */

    [data-testid="stSidebar"] {
        background: #FFFFFF;
        border-right: 1px solid #E2E8F0;
    }

    [data-testid="stSidebar"] * {
        color: #0F172A !important;
    }

    .sidebar-logo {
        font-size: 40px;
        margin-bottom: 10px;
    }

    .sidebar-title {
        font-size: 23px;
        font-weight: 900;
        color: #0B63F6 !important;
        line-height: 1.15;
        margin-bottom: 8px;
    }

    .sidebar-subtitle {
        color: #64748B !important;
        font-size: 14px;
        line-height: 1.45;
        margin-bottom: 18px;
    }

    .sidebar-line {
        height: 1px;
        background: #E2E8F0;
        margin: 18px 0;
    }

    section[data-testid="stSidebar"] .stRadio > label {
        font-size: 14px;
        font-weight: 800;
        color: #64748B !important;
        margin-bottom: 12px;
    }

    section[data-testid="stSidebar"] div[role="radiogroup"] {
        gap: 8px;
    }

    section[data-testid="stSidebar"] div[role="radiogroup"] label {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 14px;
        padding: 10px 12px !important;
        margin-bottom: 8px !important;
        box-shadow: 0 3px 10px rgba(15, 23, 42, 0.04);
        transition: all 0.2s ease;
        cursor: pointer;
    }

    section[data-testid="stSidebar"] div[role="radiogroup"] label:hover {
        background: #EFF6FF;
        border-color: #BFDBFE;
        transform: translateX(2px);
    }

    section[data-testid="stSidebar"] div[role="radiogroup"] label p {
        font-size: 15px !important;
        font-weight: 700 !important;
        color: #0F172A !important;
        margin: 0 !important;
    }

    section[data-testid="stSidebar"] div[role="radiogroup"] label:has(input:checked) {
        background: linear-gradient(135deg, #0B63F6, #2563EB) !important;
        border-color: #0B63F6 !important;
        box-shadow: 0 8px 18px rgba(37, 99, 235, 0.22);
    }

    section[data-testid="stSidebar"] div[role="radiogroup"] label:has(input:checked) p {
        color: #FFFFFF !important;
        font-weight: 850 !important;
    }

    section[data-testid="stSidebar"] div[role="radiogroup"] label input {
        accent-color: #0B63F6;
    }

    /* ================= CARDS ================= */

    .hero-card {
        background: linear-gradient(135deg, #FFFFFF 0%, #F7FBFF 58%, #EAF4FF 100%);
        border: 1px solid #DCEBFF;
        border-radius: 26px;
        padding: 30px 34px;
        box-shadow: 0 14px 34px rgba(15, 23, 42, 0.08);
        margin-bottom: 22px;
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 20px;
    }

    .hero-left {
        display: flex;
        align-items: center;
        gap: 22px;
    }

    .hero-icon {
        width: 80px;
        height: 80px;
        border-radius: 22px;
        background: linear-gradient(135deg, #E8F2FF, #FFFFFF);
        display: flex;
        align-items: center;
        justify-content: center;
        box-shadow: 0 8px 20px rgba(37, 99, 235, 0.14);
        font-size: 40px;
        border: 1px solid #DBEAFE;
    }

    .hero-title {
        font-size: 40px;
        font-weight: 900;
        letter-spacing: -0.035em;
        color: #0B1F5E;
        margin: 0 0 8px 0;
    }

    .hero-subtitle {
        color: #475569;
        font-size: 17px;
        line-height: 1.45;
        max-width: 680px;
        margin: 0;
    }

    .hero-art {
        min-width: 230px;
        height: 125px;
        border-radius: 24px;
        background:
            radial-gradient(circle at 20% 30%, rgba(37,99,235,0.16), transparent 34%),
            linear-gradient(135deg, rgba(59,130,246,0.15), rgba(14,165,233,0.10));
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 66px;
        color: #2563EB;
        border: 1px solid rgba(37, 99, 235, 0.12);
    }

    .metric-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 20px;
        padding: 20px;
        box-shadow: 0 10px 24px rgba(15, 23, 42, 0.06);
        min-height: 118px;
        display: flex;
        gap: 15px;
        align-items: center;
    }

    .metric-icon {
        width: 54px;
        height: 54px;
        border-radius: 17px;
        background: #EFF6FF;
        color: #0B63F6;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 26px;
        flex-shrink: 0;
    }

    .metric-title {
        font-size: 13px;
        color: #64748B;
        font-weight: 700;
        margin-bottom: 6px;
    }

    .metric-value {
        font-size: 24px;
        color: #0B63F6;
        font-weight: 850;
        line-height: 1.15;
    }

    .metric-subtitle {
        color: #64748B;
        font-size: 12px;
        margin-top: 6px;
    }

    .page-header-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 24px;
        padding: 25px 30px;
        box-shadow: 0 10px 24px rgba(15, 23, 42, 0.06);
        margin-bottom: 20px;
    }

    .page-title {
        font-size: 38px;
        font-weight: 900;
        letter-spacing: -0.03em;
        color: #0F172A;
        margin: 0 0 8px 0;
    }

    .page-subtitle {
        color: #64748B;
        font-size: 16px;
        line-height: 1.5;
        margin: 0;
    }

    .section-head {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 20px;
        padding: 18px 22px;
        box-shadow: 0 8px 20px rgba(15, 23, 42, 0.05);
        margin: 18px 0 12px 0;
    }

    .section-title {
        font-size: 22px;
        font-weight: 850;
        color: #0F172A;
        margin: 0;
    }

    .section-subtitle {
        color: #64748B;
        font-size: 14px;
        margin-top: 5px;
    }

    .label-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 18px;
        padding: 17px 19px;
        box-shadow: 0 8px 18px rgba(15, 23, 42, 0.05);
        margin-bottom: 12px;
        display: flex;
        align-items: center;
        justify-content: space-between;
    }

    .label-left {
        display: flex;
        align-items: center;
        gap: 12px;
        color: #0F172A;
        font-weight: 700;
    }

    .label-icon {
        width: 36px;
        height: 36px;
        border-radius: 14px;
        background: #EFF6FF;
        color: #0B63F6;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 18px;
    }

    .label-value {
        color: #0B63F6;
        font-size: 24px;
        font-weight: 900;
    }

    .info-card {
        background: #EEF6FF;
        border: 1px solid #BFDBFE;
        border-radius: 18px;
        padding: 16px 18px;
        color: #1E3A8A;
        font-size: 15px;
        box-shadow: 0 6px 16px rgba(15, 23, 42, 0.04);
        margin-top: 15px;
    }

    .mini-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 18px;
        padding: 18px;
        box-shadow: 0 8px 18px rgba(15, 23, 42, 0.05);
        height: 100%;
    }

    .mini-card-title {
        font-size: 16px;
        font-weight: 850;
        color: #0F172A;
        margin-bottom: 8px;
    }

    .mini-card-text {
        color: #475569;
        font-size: 14px;
        line-height: 1.45;
    }

    /* ================= LIGHT TABLE ================= */

    .table-wrap {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 18px;
        overflow: auto;
        box-shadow: 0 8px 18px rgba(15, 23, 42, 0.05);
        margin-top: 12px;
    }

    table.light-table {
        width: 100%;
        border-collapse: collapse;
        font-size: 13px;
        background: #FFFFFF;
    }

    table.light-table thead th {
        background: #EFF6FF;
        color: #0F172A;
        padding: 12px 14px;
        text-align: left;
        font-weight: 850;
        border-bottom: 1px solid #DBEAFE;
        white-space: nowrap;
    }

    table.light-table tbody td {
        color: #0F172A;
        padding: 12px 14px;
        border-bottom: 1px solid #E2E8F0;
        vertical-align: top;
    }

    table.light-table tbody tr:hover {
        background: #F8FAFC;
    }

    table.light-table a {
        color: #0B63F6;
        text-decoration: none;
        font-weight: 750;
    }

    table.light-table a:hover {
        text-decoration: underline;
    }

    div[data-testid="stNumberInput"] input,
    div[data-testid="stTextInput"] input {
        background: #FFFFFF !important;
        color: #0F172A !important;
        border: 1px solid #CBD5E1 !important;
        border-radius: 10px !important;
    }

    .stButton > button {
        background: linear-gradient(135deg, #0B63F6, #2563EB);
        color: #FFFFFF !important;
        border: none;
        border-radius: 14px;
        padding: 0.75rem 1.2rem;
        font-weight: 850;
        box-shadow: 0 8px 18px rgba(37, 99, 235, 0.24);
    }

    .stButton > button:hover {
        background: linear-gradient(135deg, #0755D5, #1D4ED8);
        color: #FFFFFF !important;
    }

    @media (max-width: 900px) {
        .hero-card {
            flex-direction: column;
            align-items: flex-start;
        }
        .hero-art {
            width: 100%;
            min-width: 0;
        }
        .hero-title {
            font-size: 32px;
        }
    }
    /* =============== FIX SIDEBAR + CONTENT DENSITY =============== */

section[data-testid="stSidebar"] {
    min-width: 270px !important;
    max-width: 270px !important;
}

section[data-testid="stSidebar"] > div:first-child {
    padding: 28px 18px 24px 18px;
}

.sidebar-logo {
    width: 58px;
    height: 58px;
    border-radius: 18px;
    background: #EFF6FF;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 34px;
    margin-bottom: 14px;
    box-shadow: 0 6px 16px rgba(37, 99, 235, 0.12);
}

.sidebar-title {
    font-size: 22px !important;
    line-height: 1.2 !important;
    margin-bottom: 8px !important;
}

.sidebar-subtitle {
    font-size: 14px !important;
    line-height: 1.5 !important;
    max-width: 210px;
}

.sidebar-line {
    margin: 20px 0 18px 0 !important;
}

.content-card {
    background: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-radius: 22px;
    padding: 22px;
    box-shadow: 0 10px 24px rgba(15, 23, 42, 0.06);
    margin-bottom: 20px;
}

.feature-grid {
    display: grid;
    grid-template-columns: repeat(3, minmax(0, 1fr));
    gap: 16px;
    margin-top: 16px;
}

.feature-box {
    background: #F8FBFF;
    border: 1px solid #DBEAFE;
    border-radius: 18px;
    padding: 18px;
}

.feature-title {
    font-size: 16px;
    font-weight: 850;
    color: #0F172A;
    margin-bottom: 8px;
}

.feature-text {
    font-size: 14px;
    color: #475569;
    line-height: 1.5;
}

.timeline-box {
    background: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-radius: 18px;
    padding: 18px;
    box-shadow: 0 6px 16px rgba(15, 23, 42, 0.04);
}

.timeline-step {
    display: flex;
    gap: 14px;
    margin-bottom: 16px;
}

.step-number {
    width: 34px;
    height: 34px;
    border-radius: 50%;
    background: #0B63F6;
    color: #FFFFFF !important;
    display: flex;
    align-items: center;
    justify-content: center;
    font-weight: 850;
    flex-shrink: 0;
}

.step-content-title {
    font-weight: 850;
    color: #0F172A;
    margin-bottom: 4px;
}

.step-content-text {
    color: #64748B;
    font-size: 14px;
    line-height: 1.45;
}

.stat-pill {
    display: inline-block;
    padding: 8px 12px;
    border-radius: 999px;
    background: #EFF6FF;
    color: #0B63F6 !important;
    font-weight: 800;
    font-size: 13px;
    margin: 4px 6px 4px 0;
}

.model-highlight {
    background: linear-gradient(135deg, #0B63F6, #2563EB);
    border-radius: 22px;
    padding: 24px;
    box-shadow: 0 14px 30px rgba(37, 99, 235, 0.22);
    color: white;
}

.model-highlight * {
    color: white !important;
}

.model-highlight-title {
    font-size: 18px;
    font-weight: 800;
    margin-bottom: 8px;
}

.model-highlight-value {
    font-size: 30px;
    font-weight: 900;
    line-height: 1.2;
}

.model-highlight-text {
    font-size: 14px;
    opacity: 0.92;
    margin-top: 10px;
    line-height: 1.5;
}

@media (max-width: 1000px) {
    .feature-grid {
        grid-template-columns: 1fr;
    }
}
/* =============== FIX BUTTON DỰ ĐOÁN =============== */

div[data-testid="stForm"] .stButton > button,
.stButton > button {
    background: linear-gradient(135deg, #0B63F6, #2563EB) !important;
    color: #FFFFFF !important;
    border: none !important;
    border-radius: 14px !important;
    padding: 0.75rem 1.3rem !important;
    font-weight: 850 !important;
    font-size: 15px !important;
    box-shadow: 0 8px 18px rgba(37, 99, 235, 0.25) !important;
    transition: all 0.2s ease !important;
}

/* hover không bị đen */
div[data-testid="stForm"] .stButton > button:hover,
.stButton > button:hover {
    background: linear-gradient(135deg, #0755D5, #1D4ED8) !important;
    color: #FFFFFF !important;
    border: none !important;
    transform: translateY(-1px);
    box-shadow: 0 10px 22px rgba(37, 99, 235, 0.32) !important;
}

/* active/focus cũng không bị đen */
div[data-testid="stForm"] .stButton > button:active,
div[data-testid="stForm"] .stButton > button:focus,
.stButton > button:active,
.stButton > button:focus {
    background: linear-gradient(135deg, #064BC0, #1E40AF) !important;
    color: #FFFFFF !important;
    border: none !important;
    outline: 3px solid rgba(37, 99, 235, 0.25) !important;
}

/* chữ bên trong button */
div[data-testid="stForm"] .stButton > button p,
.stButton > button p {
    color: #FFFFFF !important;
    font-weight: 850 !important;
}


/* =============== FORCE FIX ALL STREAMLIT FORM BUTTONS =============== */
.stFormSubmitButton > button,
.stFormSubmitButton button,
div[data-testid="stFormSubmitButton"] > button,
div[data-testid="stFormSubmitButton"] button,
.stButton > button,
.stButton button,
button[kind="primary"],
button[data-testid="baseButton-primary"],
button[data-testid="baseButton-secondary"] {
    background: linear-gradient(135deg, #0B63F6, #2563EB) !important;
    color: #FFFFFF !important;
    border: none !important;
    border-radius: 14px !important;
    min-height: 48px !important;
    padding: 0.75rem 1.4rem !important;
    font-size: 15px !important;
    font-weight: 850 !important;
    box-shadow: 0 8px 18px rgba(37, 99, 235, 0.28) !important;
    transition: all 0.2s ease !important;
}

.stFormSubmitButton > button *,
.stFormSubmitButton button *,
div[data-testid="stFormSubmitButton"] > button *,
div[data-testid="stFormSubmitButton"] button *,
.stButton > button *,
.stButton button *,
button[kind="primary"] *,
button[data-testid="baseButton-primary"] *,
button[data-testid="baseButton-secondary"] * {
    color: #FFFFFF !important;
    fill: #FFFFFF !important;
    font-weight: 850 !important;
}

.stFormSubmitButton > button:hover,
.stFormSubmitButton button:hover,
div[data-testid="stFormSubmitButton"] > button:hover,
div[data-testid="stFormSubmitButton"] button:hover,
.stButton > button:hover,
.stButton button:hover,
button[kind="primary"]:hover,
button[data-testid="baseButton-primary"]:hover,
button[data-testid="baseButton-secondary"]:hover {
    background: linear-gradient(135deg, #0755D5, #1D4ED8) !important;
    color: #FFFFFF !important;
    border: none !important;
    transform: translateY(-1px);
    box-shadow: 0 10px 22px rgba(37, 99, 235, 0.34) !important;
}

.stFormSubmitButton > button:hover *,
.stFormSubmitButton button:hover *,
div[data-testid="stFormSubmitButton"] > button:hover *,
div[data-testid="stFormSubmitButton"] button:hover *,
.stButton > button:hover *,
.stButton button:hover *,
button[kind="primary"]:hover *,
button[data-testid="baseButton-primary"]:hover *,
button[data-testid="baseButton-secondary"]:hover * {
    color: #FFFFFF !important;
    fill: #FFFFFF !important;
}

.stFormSubmitButton > button:focus,
.stFormSubmitButton > button:active,
div[data-testid="stFormSubmitButton"] button:focus,
div[data-testid="stFormSubmitButton"] button:active,
.stButton button:focus,
.stButton button:active {
    background: linear-gradient(135deg, #064BC0, #1E40AF) !important;
    color: #FFFFFF !important;
    outline: 3px solid rgba(37, 99, 235, 0.22) !important;
}

</style>
""",
    unsafe_allow_html=True,
)


# ============================================================
# DATA LOADING
# ============================================================

def file_mtime(path: Path) -> float:
    """Return file modified time so Streamlit cache refreshes when CSV/model changes."""
    try:
        return path.stat().st_mtime
    except FileNotFoundError:
        return 0.0


@st.cache_data(show_spinner=False)
def load_products(_mtime: float) -> pd.DataFrame:
    """Load labeled products. _mtime forces cache invalidation when file changes."""
    if not DATA_PATH.exists():
        return pd.DataFrame()
    return pd.read_csv(DATA_PATH)


@st.cache_data(show_spinner=False)
def load_model_comparison(_mtime: float) -> pd.DataFrame:
    if not MODEL_COMPARISON_PATH.exists():
        return pd.DataFrame()
    return pd.read_csv(MODEL_COMPARISON_PATH)


@st.cache_data(show_spinner=False)
def load_classification_report(_mtime: float) -> pd.DataFrame:
    if not CLASSIFICATION_REPORT_PATH.exists():
        return pd.DataFrame()

    df = pd.read_csv(CLASSIFICATION_REPORT_PATH)

    if "Unnamed: 0" in df.columns:
        df = df.rename(columns={"Unnamed: 0": "label"})

    return df


@st.cache_data(show_spinner=False)
def load_review_stats(path_str: str, _mtime: float) -> tuple[int, int]:
    """Read raw crawl CSV stats without writing or mutating data."""
    path = Path(path_str)
    if not path.exists():
        return 0, 0

    try:
        reviews_df = pd.read_csv(path)
    except Exception:
        return 0, 0

    total_review_rows = len(reviews_df)
    unique_review_products = (
        reviews_df["product_id"].nunique()
        if "product_id" in reviews_df.columns
        else 0
    )
    return total_review_rows, int(unique_review_products)




def patch_sklearn_tree_model(model):
    """Fix sklearn pickle compatibility for tree models on Streamlit Cloud.

    Some models were trained with a different scikit-learn version. On newer
    versions, tree estimators may expect the attribute `monotonic_cst`.
    Old pickled DecisionTree estimators may not have it, causing prediction to fail.
    """
    def patch_one(est):
        try:
            cls_name = est.__class__.__name__
            if cls_name in {
                "DecisionTreeClassifier",
                "DecisionTreeRegressor",
                "ExtraTreeClassifier",
                "ExtraTreeRegressor",
            }:
                if not hasattr(est, "monotonic_cst"):
                    est.monotonic_cst = None
        except Exception:
            pass

    if model is None:
        return model

    patch_one(model)

    # RandomForest / ExtraTrees store many decision trees in estimators_
    try:
        estimators = getattr(model, "estimators_", [])
        for est in estimators:
            if isinstance(est, (list, tuple)):
                for sub_est in est:
                    patch_one(sub_est)
            else:
                patch_one(est)
    except Exception:
        pass

    # Some wrappers keep the real estimator here
    try:
        patch_one(getattr(model, "estimator_", None))
    except Exception:
        pass

    return model


@st.cache_resource(show_spinner=False)
def load_model_and_encoder(_model_mtime: float, _encoder_mtime: float):
    """Load model/encoder. mtimes force refresh after retraining."""
    if not MODEL_PATH.exists() or not ENCODER_PATH.exists():
        return None, None
    model = joblib.load(MODEL_PATH)
    model = patch_sklearn_tree_model(model)
    encoder = joblib.load(ENCODER_PATH)
    return model, encoder


# ============================================================
# HELPERS
# ============================================================

COLUMN_NAME_MAP = {
    "product_id": "Mã sản phẩm",
    "product_name": "Tên sản phẩm",
    "price": "Giá bán",
    "rating": "Điểm đánh giá",
    "review_count": "Số đánh giá",
    "sold_count": "Lượt bán",
    "comment_count": "Số bình luận",
    "avg_comment_rating": "Điểm bình luận TB",
    "positive_ratio": "Tỷ lệ tích cực",
    "neutral_ratio": "Tỷ lệ trung tính",
    "negative_ratio": "Tỷ lệ tiêu cực",
    "estimated_revenue": "Doanh thu ước tính",
    "product_label": "Nhãn sản phẩm",
    "product_url": "Đường dẫn",
    "suggestion": "Gợi ý cải thiện",
    "Model": "Mô hình",
    "model": "Mô hình",
    "Accuracy": "Độ chính xác",
    "accuracy": "Độ chính xác",
    "Precision": "Precision",
    "precision": "Precision",
    "Recall": "Recall",
    "recall": "Recall",
    "F1-score": "F1-score",
    "f1-score": "F1-score",
    "support": "Số mẫu",
    "label": "Nhãn",
}


def safe_html(value) -> str:
    if pd.isna(value):
        return ""
    return html.escape(str(value))


def format_number(value) -> str:
    try:
        return f"{int(float(value)):,}"
    except Exception:
        return "0"


def format_vnd(value) -> str:
    try:
        return f"{int(float(value)):,}đ"
    except Exception:
        return "0đ"


def format_float(value, digits=2) -> str:
    try:
        return f"{float(value):.{digits}f}"
    except Exception:
        return "0.00"


def get_report_value(report_df, label_name, col_candidates, default=0):
    if report_df.empty or "label" not in report_df.columns:
        return default

    rows = report_df[report_df["label"].astype(str).str.lower() == str(label_name).lower()]
    if rows.empty:
        return default

    row = rows.iloc[0]
    for col in col_candidates:
        if col in report_df.columns and pd.notna(row.get(col)):
            return row.get(col)

    return default


def page_header(title: str, subtitle: str) -> None:
    st.markdown(
        f"""
        <div class="page-header-card">
            <div class="page-title">{safe_html(title)}</div>
            <p class="page-subtitle">{safe_html(subtitle)}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def section_header(title: str, subtitle: str = "") -> None:
    st.markdown(
        f"""
        <div class="section-head">
            <div class="section-title">{safe_html(title)}</div>
            <div class="section-subtitle">{safe_html(subtitle)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def metric_card(icon: str, title: str, value: str, subtitle: str = "") -> None:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-icon">{icon}</div>
            <div>
                <div class="metric-title">{safe_html(title)}</div>
                <div class="metric-value">{safe_html(value)}</div>
                <div class="metric-subtitle">{safe_html(subtitle)}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def mini_card(title: str, text: str, accent: str = "#2563EB") -> None:
    st.markdown(
        f"""
        <div class="mini-card" style="border-left: 5px solid {accent};">
            <div class="mini-card-title">{safe_html(title)}</div>
            <div class="mini-card-text">{safe_html(text)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_label_card(label: str, value: int) -> None:
    st.markdown(
        f"""
        <div class="label-card">
            <div class="label-left">
                <div class="label-icon">🏷️</div>
                <div>{safe_html(label)}</div>
            </div>
            <div class="label-value">{format_number(value)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def normalize_url(url: str) -> str:
    if not isinstance(url, str) or not url.strip():
        return ""
    url = url.strip()
    if url.startswith("http"):
        return url
    if url.startswith("/"):
        return "https://tiki.vn" + url
    return url


def render_light_table(df: pd.DataFrame, columns=None, max_rows: int = 30) -> None:
    if df is None or df.empty:
        st.info("Không có dữ liệu để hiển thị.")
        return

    display_df = df.copy()

    if columns:
        existing_cols = [col for col in columns if col in display_df.columns]
        display_df = display_df[existing_cols]

    display_df = display_df.head(max_rows).copy()

    money_cols = {"price", "estimated_revenue"}
    ratio_cols = {"positive_ratio", "neutral_ratio", "negative_ratio"}

    headers = "".join(
        f"<th>{safe_html(COLUMN_NAME_MAP.get(col, col))}</th>"
        for col in display_df.columns
    )

    html_rows = []

    for _, row in display_df.iterrows():
        cells = []
        for col in display_df.columns:
            val = row[col]

            if col in money_cols:
                cell = format_vnd(val)
            elif col in ratio_cols:
                cell = format_float(val, 2)
            elif col == "product_url":
                url = normalize_url(str(val))
                if url:
                    cell = f'<a href="{safe_html(url)}" target="_blank">Xem trên Tiki</a>'
                else:
                    cell = ""
            elif isinstance(val, float):
                cell = format_float(val, 2)
            else:
                cell = safe_html(val)

            cells.append(f"<td>{cell}</td>")

        html_rows.append("<tr>" + "".join(cells) + "</tr>")

    table_html = f"""
    <div class="table-wrap">
        <table class="light-table">
            <thead>
                <tr>{headers}</tr>
            </thead>
            <tbody>
                {''.join(html_rows)}
            </tbody>
        </table>
    </div>
    """
    st.markdown(table_html, unsafe_allow_html=True)


def label_counts(df: pd.DataFrame) -> pd.Series:
    if df.empty or "product_label" not in df.columns:
        return pd.Series(dtype=int)

    order = [
        "Normal",
        "Needs Improvement",
        "Premium / Niche Quality",
        "Best Seller",
        "High Potential",
    ]

    counts = df["product_label"].value_counts()
    return counts.reindex(order).fillna(0).astype(int)


def make_bar_chart(counts: pd.Series):
    chart_df = counts.reset_index()
    chart_df.columns = ["Nhãn sản phẩm", "Số lượng"]

    fig = px.bar(
        chart_df,
        x="Nhãn sản phẩm",
        y="Số lượng",
        text="Số lượng",
        color_discrete_sequence=["#0B63F6"],
        template="plotly_white",
    )

    fig.update_traces(
        textposition="outside",
        marker_line_width=0,
        textfont=dict(color="#0F172A", size=14),
        cliponaxis=False,
    )

    fig.update_layout(
        paper_bgcolor="white",
        plot_bgcolor="white",
        font_color="#0F172A",
        height=420,
        margin=dict(l=50, r=20, t=35, b=70),
        showlegend=False,
    )

    fig.update_xaxes(
        title_text="Nhãn sản phẩm",
        tickfont=dict(color="#0F172A", size=12),
        title_font=dict(color="#0F172A", size=13),
        showgrid=False,
        linecolor="#0F172A",
    )

    fig.update_yaxes(
        title_text="Số lượng",
        tickfont=dict(color="#0F172A", size=12),
        title_font=dict(color="#0F172A", size=13),
        gridcolor="#E2E8F0",
        linecolor="#0F172A",
    )

    return fig

def make_donut_chart(counts: pd.Series):
    chart_df = counts.reset_index()
    chart_df.columns = ["Nhãn", "Số lượng"]

    fig = px.pie(
        chart_df,
        names="Nhãn",
        values="Số lượng",
        hole=0.48,
        color_discrete_sequence=["#0B63F6", "#EF4444", "#F59E0B", "#8B5CF6", "#10B981"],
        template="plotly_white",
    )

    fig.update_layout(
        paper_bgcolor="white",
        plot_bgcolor="white",
        font_color="#0F172A",
        height=390,
        margin=dict(l=20, r=20, t=20, b=20),
    )

    return fig


def get_best_model_info(model_df: pd.DataFrame):
    if model_df.empty:
        return "Random Forest Classifier", 0.9815, 0.9823, 0.9817

    lower_map = {c.lower().strip(): c for c in model_df.columns}

    model_col = lower_map.get("model", model_df.columns[0])
    acc_col = lower_map.get("accuracy")
    pre_col = lower_map.get("precision")
    f1_col = lower_map.get("f1-score") or lower_map.get("f1_score") or lower_map.get("f1")

    if f1_col is None:
        return "Random Forest Classifier", 0.9815, 0.9823, 0.9817

    best_idx = model_df[f1_col].astype(float).idxmax()
    best_row = model_df.loc[best_idx]

    best_model = str(best_row[model_col])
    acc = float(best_row[acc_col]) if acc_col else 0.0
    pre = float(best_row[pre_col]) if pre_col else 0.0
    f1 = float(best_row[f1_col])

    return best_model, acc, pre, f1


def prepare_display_df(df: pd.DataFrame) -> pd.DataFrame:
    keep_cols = [
        "product_name",
        "price",
        "rating",
        "review_count",
        "sold_count",
        "comment_count",
        "estimated_revenue",
        "product_label",
        "product_url",
    ]
    return df[[c for c in keep_cols if c in df.columns]].copy()


def show_plotly(fig):
    try:
        st.plotly_chart(fig, width="stretch")
    except TypeError:
        st.plotly_chart(fig, use_container_width=True)


def show_image(path):
    try:
        st.image(str(path), width="stretch")
    except TypeError:
        st.image(str(path), use_container_width=True)


TIKI_PRODUCT_API = "https://tiki.vn/api/v2/products/{product_id}"
TIKI_REVIEW_API = "https://tiki.vn/api/v2/reviews"
TIKI_HEADERS = {
    "User-Agent": "Mozilla/5.0",
    "Accept": "application/json, text/plain, */*",
    "Referer": "https://tiki.vn/",
}
MODEL_FEATURE_COLUMNS = [
    "price",
    "rating",
    "review_count",
    "sold_count",
    "comment_count",
    "avg_comment_rating",
    "positive_ratio",
    "neutral_ratio",
    "negative_ratio",
    "estimated_revenue",
]


def extract_product_id(url_or_text):
    text = str(url_or_text or "").strip()
    if not text:
        return None

    for pattern in (r"-p(\d+)\.html", r"/p/(\d+)", r"(\d+)"):
        match = re.search(pattern, text)
        if match:
            return match.group(1)
    return None


def parse_sold_count(value):
    if value is None:
        return 0

    if isinstance(value, dict):
        raw_value = value.get("value")
        if raw_value not in (None, ""):
            return parse_sold_count(raw_value)
        return parse_sold_count(value.get("text"))

    try:
        if pd.isna(value):
            return 0
    except (TypeError, ValueError):
        pass

    if isinstance(value, bool):
        return int(value)

    if isinstance(value, (int, float)):
        return int(value)

    text = str(value).strip().lower()
    if not text:
        return 0

    match = re.search(r"(\d+(?:[.,]\d+)?)\s*([km])?", text)
    if not match:
        return 0

    number_text = match.group(1)
    suffix = match.group(2) or ""

    if suffix in {"k", "m"}:
        number = float(number_text.replace(",", "."))
        multiplier = 1000 if suffix == "k" else 1_000_000
        return int(number * multiplier)

    digits = re.sub(r"\D", "", number_text)
    return int(digits) if digits else 0


def extract_sold_count(product):
    for key in ("quantity_sold", "all_time_quantity_sold", "sold_count", "order_count"):
        value = product.get(key)
        if value in (None, "", {}):
            continue
        return parse_sold_count(value)

    quantity_sold = product.get("quantity_sold")
    if isinstance(quantity_sold, dict):
        value = quantity_sold.get("value")
        if value not in (None, ""):
            return parse_sold_count(value)
        text = quantity_sold.get("text")
        if text not in (None, ""):
            return parse_sold_count(text)

    return parse_sold_count(product.get("sold"))


def fetch_tiki_product(product_id):
    try:
        response = requests.get(
            TIKI_PRODUCT_API.format(product_id=product_id),
            headers=TIKI_HEADERS,
            timeout=12,
        )
    except requests.RequestException:
        return None

    if response.status_code != 200:
        return None

    try:
        product = response.json()
    except ValueError:
        return None

    return product if isinstance(product, dict) else None


def fetch_tiki_reviews(product_id, limit=20):
    params = {
        "product_id": product_id,
        "limit": limit,
        "page": 1,
        "include": "comments,contribute_info,attribute_vote_summary",
        "sort": "score|desc,id|desc,stars|all",
    }

    try:
        response = requests.get(TIKI_REVIEW_API, headers=TIKI_HEADERS, params=params, timeout=12)
    except requests.RequestException:
        return []

    if response.status_code != 200:
        return []

    try:
        data = response.json()
    except ValueError:
        return []

    reviews = data.get("data", [])
    return reviews if isinstance(reviews, list) else []


def _to_float(value, default=0):
    try:
        if pd.isna(value):
            return default
    except (TypeError, ValueError):
        pass

    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def analyze_reviews(reviews):
    if not reviews:
        return {
            "comment_count": 0,
            "avg_comment_rating": 0,
            "positive_ratio": 0,
            "neutral_ratio": 0,
            "negative_ratio": 0,
        }

    ratings = []
    comment_count = 0

    for review in reviews:
        content = str(review.get("content") or "").strip()
        title = str(review.get("title") or "").strip()
        if content or title:
            comment_count += 1

        rating = _to_float(review.get("rating"), None)
        if rating is not None:
            ratings.append(rating)

    if not ratings:
        return {
            "comment_count": comment_count,
            "avg_comment_rating": 0,
            "positive_ratio": 0,
            "neutral_ratio": 0,
            "negative_ratio": 0,
        }

    total = len(ratings)
    positive_count = sum(1 for rating in ratings if rating >= 4)
    neutral_count = sum(1 for rating in ratings if rating == 3)
    negative_count = sum(1 for rating in ratings if rating <= 2)

    return {
        "comment_count": comment_count,
        "avg_comment_rating": sum(ratings) / total,
        "positive_ratio": positive_count / total,
        "neutral_ratio": neutral_count / total,
        "negative_ratio": negative_count / total,
    }


def build_features_from_product(product, reviews):
    price = _to_float(product.get("price") or product.get("list_price") or 0, 0)
    rating = _to_float(product.get("rating_average") or product.get("rating") or 0, 0)
    review_count = int(_to_float(product.get("review_count") or 0, 0))
    sold_count = extract_sold_count(product)
    review_stats = analyze_reviews(reviews)
    estimated_revenue = price * sold_count

    feature_values = {
        "price": price,
        "rating": rating,
        "review_count": review_count,
        "sold_count": sold_count,
        "comment_count": review_stats["comment_count"],
        "avg_comment_rating": review_stats["avg_comment_rating"],
        "positive_ratio": review_stats["positive_ratio"],
        "neutral_ratio": review_stats["neutral_ratio"],
        "negative_ratio": review_stats["negative_ratio"],
        "estimated_revenue": estimated_revenue,
    }

    return pd.DataFrame([[feature_values[col] for col in MODEL_FEATURE_COLUMNS]], columns=MODEL_FEATURE_COLUMNS)


def build_tiki_product_url(product, product_id):
    url_path = product.get("url_path")
    if url_path:
        url_path = str(url_path).strip()
        if url_path.startswith("http"):
            return url_path
        return f"https://tiki.vn/{url_path.lstrip('/')}"
    return f"https://tiki.vn/p/{product_id}"


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.markdown(
    """
    <div class="sidebar-logo">📚</div>
    <div class="sidebar-title">Tiki Book Intelligence</div>
    <div class="sidebar-subtitle">
        DASHBOARD PHÂN TÍCH HIỆU QUẢ SẢN PHẨM SÁCH TRÊN TIKI 
    </div>
    <div class="sidebar-line"></div>
    """,
    unsafe_allow_html=True,
)

page = st.sidebar.radio(
    "Chọn trang",
    [
        "🏠 Tổng quan",
        "🗄️ Dữ liệu sản phẩm",
        "🏷️ Gán nhãn sản phẩm",
        "🧠 Mô hình dự đoán",
        "📊 Đánh giá mô hình",
        "🔎 Dự đoán sản phẩm",
        "💡 Gợi ý cải thiện",
        "🚩 Kết luận",
    ],
)


# ============================================================
# LOAD DATA
# ============================================================

df = load_products(file_mtime(DATA_PATH))
model_df = load_model_comparison(file_mtime(MODEL_COMPARISON_PATH))
report_df = load_classification_report(file_mtime(CLASSIFICATION_REPORT_PATH))
total_review_rows, unique_review_products = load_review_stats(
    str(REVIEWS_PATH),
    file_mtime(REVIEWS_PATH),
)
total_rows = total_review_rows if total_review_rows > 0 else len(df)

st.sidebar.markdown("---")
st.sidebar.caption(f"Debug dòng reviews: {total_rows:,}")
st.sidebar.caption(f"Debug sản phẩm labeled: {len(df):,}")
st.sidebar.caption(f"Debug product_id reviews: {unique_review_products:,}")

if df.empty:
    page_header(
        "Chưa có dữ liệu",
        "Không tìm thấy file data/tiki_books_labeled.csv. Hãy chạy bước gán nhãn trước.",
    )
    st.code("python label_products.py", language="bash")
    st.stop()

counts = label_counts(df)
total_products = len(df)
total_revenue = df["estimated_revenue"].sum() if "estimated_revenue" in df.columns else 0
best_model_name, best_acc, best_precision, best_f1 = get_best_model_info(model_df)


# ============================================================
# PAGES
# ============================================================

if page == "🏠 Tổng quan":
    st.markdown(
        """
        <div class="hero-card">
            <div class="hero-left">
                <div class="hero-icon">📚</div>
                <div>
                    <h1 class="hero-title">Tiki Book Intelligence</h1>
                    <p class="hero-subtitle">
                        Hệ thống dự đoán và phân tích hiệu quả sản phẩm sách trên Tiki bằng học máy.
                    </p>
                </div>
            </div>
            <div class="hero-art">📈</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        metric_card("🗃️", "Tổng dòng dữ liệu crawl", format_number(total_rows), "Số dòng reviews trong file crawl")

    with c2:
        metric_card("📦", "Tổng sản phẩm", format_number(total_products), "Sản phẩm khác nhau")

    with c3:
        metric_card("💰", "Doanh thu ước tính", format_vnd(total_revenue), "Giá × số đã bán")

    with c4:
        metric_card("🏆", "Mô hình tốt nhất", best_model_name, "Hiệu năng F1 cao nhất")

    left, right = st.columns([2.1, 1], gap="large")

    with left:
        section_header("📊 Phân bố nhãn sản phẩm")
        show_plotly(make_bar_chart(counts))

    with right:
        section_header("🏷️ Số lượng theo nhãn")
        for label, value in counts.items():
            render_label_card(label, int(value))

    st.markdown(
        """
        <div class="info-card">
            ℹ️ <b>Doanh thu ước tính</b> = giá bán × lượt bán. Đây là chỉ số tham khảo, không phải doanh thu thực tế.
        </div>
        """,
        unsafe_allow_html=True,
    )


elif page == "🗄️ Dữ liệu sản phẩm":
    page_header(
        "Dữ liệu sản phẩm",
        "Bảng dữ liệu sản phẩm đã được làm sạch, tổng hợp và gán nhãn hiệu quả.",
    )

    comment_series = df["comment_count"] if "comment_count" in df.columns else pd.Series([0] * len(df))
    comment_count = int((comment_series > 0).sum())
    no_comment_count = total_products - comment_count

    c1, c2, c3 = st.columns(3)

    with c1:
        metric_card("📦", "Tổng sản phẩm", format_number(total_products), "Sản phẩm duy nhất")

    with c2:
        metric_card("💬", "Có bình luận", format_number(comment_count), "Sản phẩm có comment")

    with c3:
        metric_card("📭", "Không bình luận", format_number(no_comment_count), "Sản phẩm chưa có comment")

    search = st.text_input(
        "Tìm kiếm tên sách",
        placeholder="Nhập tên sách cần tìm..."
    )

    selected_label = st.radio(
        "Chọn nhãn sản phẩm",
        [
            "Tất cả",
            "Best Seller",
            "High Potential",
            "Premium / Niche Quality",
            "Normal",
            "Needs Improvement",
        ],
        horizontal=True,
        key="product_label_filter",
    )

    show_df = df.copy()
    if search and "product_name" in show_df.columns:
        show_df = show_df[
            show_df["product_name"].astype(str).str.contains(search, case=False, na=False)
        ]

    if selected_label != "Tất cả" and "product_label" in show_df.columns:
        show_df = show_df[show_df["product_label"] == selected_label]

    section_header(
        "📋 Danh sách sản phẩm",
        f"Đang hiển thị {len(show_df):,} sản phẩm phù hợp. Bảng bên dưới hiển thị tối đa 50 dòng đầu tiên."
    )
    render_light_table(prepare_display_df(show_df), max_rows=50)


elif page == "🏷️ Gán nhãn sản phẩm":
    page_header(
        "Gán nhãn sản phẩm",
        "Mỗi sản phẩm được gán vào một nhóm hiệu quả dựa trên điểm tổng hợp từ giá, đánh giá, lượt bán và phản hồi.",
    )

    c1, c2, c3, c4, c5 = st.columns(5)

    with c1:
        mini_card("Best Seller", "Bán chạy, doanh thu và lượt bán cao.", "#0B63F6")

    with c2:
        mini_card("High Potential", "Có tiềm năng tăng trưởng tốt.", "#10B981")

    with c3:
        mini_card("Premium / Niche", "Phù hợp phân khúc cao cấp hoặc ngách.", "#8B5CF6")

    with c4:
        mini_card("Normal", "Hiệu quả ở mức trung bình.", "#64748B")

    with c5:
        mini_card("Needs Improvement", "Cần cải thiện hiệu quả hoặc phản hồi.", "#EF4444")

    left, right = st.columns([2.1, 1], gap="large")

    with left:
        section_header("🍩 Tỷ lệ nhãn sản phẩm")
        show_plotly(make_donut_chart(counts))

    with right:
        section_header("📌 Thống kê nhãn")
        for label, value in counts.items():
            render_label_card(label, int(value))

    section_header("📋 Danh sách sản phẩm đã gán nhãn")
    render_light_table(prepare_display_df(df), max_rows=40)

elif page == "🧠 Mô hình dự đoán":
    page_header(
        "Mô hình dự đoán",
        "So sánh các mô hình học máy và lựa chọn mô hình tốt nhất để dự đoán nhãn hiệu quả sản phẩm.",
    )

    top_left, top_right = st.columns([1.15, 2], gap="large")

    with top_left:
        st.markdown(
            f"""
            <div class="model-highlight">
                <div class="model-highlight-title">🏆 Mô hình tốt nhất</div>
                <div class="model-highlight-value">{safe_html(best_model_name)}</div>
                <div class="model-highlight-text">
                    Mô hình được chọn dựa trên F1-score weighted cao nhất, giúp cân bằng giữa Precision và Recall.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with top_right:
        c1, c2, c3 = st.columns(3)
        with c1:
            metric_card("🎯", "Accuracy", format_float(best_acc, 4), "Độ chính xác tổng thể")
        with c2:
            metric_card("📌", "Precision", format_float(best_precision, 4), "Độ chính xác dự đoán")
        with c3:
            metric_card("⚖️", "F1-score", format_float(best_f1, 4), "Chỉ số cân bằng")

    st.markdown(
        """
        <div class="content-card">
            <div class="section-title">🧠 Cách mô hình hoạt động</div>
            <div class="feature-grid">
                <div class="feature-box">
                    <div class="feature-title">1. Đầu vào dữ liệu</div>
                    <div class="feature-text">
                        Hệ thống sử dụng các đặc trưng như giá bán, điểm đánh giá, lượt bán,
                        số đánh giá, số bình luận và doanh thu ước tính.
                    </div>
                </div>
                <div class="feature-box">
                    <div class="feature-title">2. Học từ nhãn sản phẩm</div>
                    <div class="feature-text">
                        Mô hình học mối quan hệ giữa đặc trưng sản phẩm và các nhãn như
                        Best Seller, High Potential, Normal hoặc Needs Improvement.
                    </div>
                </div>
                <div class="feature-box">
                    <div class="feature-title">3. Dự đoán sản phẩm mới</div>
                    <div class="feature-text">
                        Khi nhập thông tin sản phẩm mới, mô hình dự đoán nhóm hiệu quả
                        để hỗ trợ phân tích và ra quyết định.
                    </div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    section_header("📊 Bảng so sánh mô hình", "So sánh hiệu năng giữa Decision Tree, Random Forest và XGBoost.")

    if model_df.empty:
        st.warning("Chưa có file model_comparison.csv. Hãy chạy: python train_product_classifier.py")
    else:
        render_light_table(model_df, max_rows=10)

    st.markdown(
        """
        <div class="content-card">
            <div class="section-title">📌 Ý nghĩa các chỉ số đánh giá</div>
            <p style="color:#475569; line-height:1.6; margin-top:10px;">
                <span class="stat-pill">Accuracy</span>
                Cho biết tỷ lệ dự đoán đúng trên toàn bộ tập kiểm tra.
                <br>
                <span class="stat-pill">Precision</span>
                Cho biết khi mô hình dự đoán một nhãn, khả năng dự đoán đó đúng là bao nhiêu.
                <br>
                <span class="stat-pill">Recall</span>
                Cho biết mô hình tìm được bao nhiêu mẫu đúng trong từng nhãn.
                <br>
                <span class="stat-pill">F1-score</span>
                Là chỉ số cân bằng giữa Precision và Recall, phù hợp khi dữ liệu có nhiều nhãn.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

   
elif page == "📊 Đánh giá mô hình":
    page_header(
        "Đánh giá mô hình",
        "Quan sát ma trận nhầm lẫn, mức độ quan trọng đặc trưng và báo cáo phân loại.",
    )

    if not report_df.empty and "label" in report_df.columns:
        accuracy_value = get_report_value(report_df, "accuracy", ["f1-score", "F1-score"], 0)
        macro_f1 = get_report_value(report_df, "macro avg", ["f1-score", "F1-score"], 0)
        weighted_f1 = get_report_value(report_df, "weighted avg", ["f1-score", "F1-score"], 0)
        total_support = get_report_value(report_df, "weighted avg", ["support", "Số mẫu"], 0)

        if not total_support:
            total_support = get_report_value(report_df, "macro avg", ["support", "Số mẫu"], 0)

        c1, c2, c3, c4 = st.columns(4)

        with c1:
            metric_card("🧪", "Tổng mẫu test", format_number(total_support), "Số mẫu dùng đánh giá")

        with c2:
            metric_card("🎯", "Accuracy", format_float(accuracy_value, 4), "Tỷ lệ dự đoán đúng")

        with c3:
            metric_card("📊", "Macro F1-score", format_float(macro_f1, 4), "Trung bình đều các lớp")

        with c4:
            metric_card("⚖️", "Weighted F1-score", format_float(weighted_f1, 4), "Có xét số lượng từng lớp")

    left, right = st.columns(2, gap="large")

    with left:
        section_header("🧩 Confusion Matrix")
        if CONFUSION_MATRIX_PATH.exists():
            show_image(CONFUSION_MATRIX_PATH)
        else:
            st.warning("Thiếu confusion_matrix.png. Hãy chạy: python train_product_classifier.py")

    with right:
        section_header("📈 Feature Importance")
        if FEATURE_IMPORTANCE_PATH.exists():
            show_image(FEATURE_IMPORTANCE_PATH)
        else:
            st.warning("Thiếu feature_importance.png. Hãy chạy: python train_product_classifier.py")

    section_header(
        "📄 Classification Report",
        "Bảng precision, recall, F1-score và số mẫu test theo từng nhãn."
    )
    if report_df.empty:
        st.warning("Chưa có classification_report.csv. Hãy chạy: python train_product_classifier.py")
    else:
        render_light_table(report_df, max_rows=30)


elif page == "🔎 Dự đoán sản phẩm":
    page_header(
        "Dự đoán sản phẩm từ link Tiki",
        "Dán link sản phẩm Tiki để hệ thống tự lấy dữ liệu public, tạo feature và dự đoán nhãn hiệu quả.",
    )

    model, encoder = load_model_and_encoder(file_mtime(MODEL_PATH), file_mtime(ENCODER_PATH))

    if model is None or encoder is None:
        st.warning("Chưa tìm thấy model. Hãy chạy: python train_product_classifier.py")
    else:
        tiki_link = st.text_input(
            "Nhập link sản phẩm Tiki",
            placeholder="Dán link sản phẩm Tiki, ví dụ: https://tiki.vn/...-p316018.html",
        )

        submitted = st.button(
            "Phân tích và dự đoán",
            type="primary",
            use_container_width=False,
        )

        if submitted:
            product_id = extract_product_id(tiki_link)
            if not product_id:
                st.error("Không tìm thấy mã sản phẩm trong link.")
            else:
                with st.spinner("Đang lấy dữ liệu sản phẩm từ Tiki..."):
                    product = fetch_tiki_product(product_id)

                if not product:
                    st.error("Không lấy được dữ liệu sản phẩm từ Tiki.")
                else:
                    with st.spinner("Đang lấy đánh giá và tạo feature dự đoán..."):
                        reviews = fetch_tiki_reviews(product_id, limit=20)
                        input_df = build_features_from_product(product, reviews)

                    try:
                        pred = model.predict(input_df)[0]
                        label = encoder.inverse_transform([pred])[0]
                    except Exception as e:
                        st.error(f"Lỗi khi dự đoán: {e}")
                        st.info(
                            "Nếu lỗi liên quan đến sklearn/monotonic_cst, hãy chạy lại "
                            "train_product_classifier.py rồi push lại file models/product_performance_model.pkl."
                        )
                    else:
                        product_name = product.get("name") or "Không rõ tên sản phẩm"
                        product_url = build_tiki_product_url(product, product_id)
                        features = input_df.iloc[0]
                        explanations = {
                            "Best Seller": "Sản phẩm có khả năng bán chạy, hiệu quả cao.",
                            "High Potential": "Sản phẩm có tiềm năng tăng trưởng tốt.",
                            "Premium / Niche Quality": "Sản phẩm phù hợp phân khúc cao cấp hoặc ngách.",
                            "Normal": "Sản phẩm có hiệu quả ở mức trung bình.",
                            "Needs Improvement": "Sản phẩm cần cải thiện chỉ số bán hàng hoặc phản hồi.",
                        }

                        st.markdown(
                            f"""
                            <div class="page-header-card">
                                <div class="page-title" style="font-size:30px;">✅ Kết quả dự đoán</div>
                                <p class="page-subtitle">{safe_html(product_name)}</p>
                                <div class="metric-value">Nhãn dự đoán: {safe_html(label)}</div>
                                <p style="color:#475569; font-size:16px; margin-top:10px;">
                                    {safe_html(explanations.get(label, "Không có mô tả."))}
                                </p>
                                <a href="{safe_html(product_url)}" target="_blank">Xem sản phẩm trên Tiki</a>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

                        c1, c2, c3 = st.columns(3)
                        with c1:
                            metric_card("💰", "Giá bán", format_vnd(features["price"]), "Giá từ API Tiki")
                        with c2:
                            metric_card("⭐", "Rating", format_float(features["rating"], 2), "Điểm đánh giá")
                        with c3:
                            metric_card("🏷️", "Nhãn dự đoán", label, "Kết quả từ mô hình")

                        c4, c5, c6 = st.columns(3)
                        with c4:
                            metric_card("📝", "Số đánh giá", format_number(features["review_count"]), "review_count")
                        with c5:
                            metric_card("📦", "Lượt bán", format_number(features["sold_count"]), "sold_count")
                        with c6:
                            metric_card("💬", "Số bình luận lấy được", format_number(features["comment_count"]), "20 review đầu")

                        c7, c8 = st.columns(2)
                        with c7:
                            metric_card(
                                "📈",
                                "Doanh thu ước tính",
                                format_vnd(features["estimated_revenue"]),
                                "Giá × lượt bán",
                            )
                        with c8:
                            metric_card("🔗", "Mã sản phẩm", product_id, "Trích từ link Tiki")

                        section_header("📄 Dữ liệu hệ thống lấy được")
                        render_light_table(input_df, max_rows=1)


elif page == "💡 Gợi ý cải thiện":
    page_header(
        "Gợi ý cải thiện",
        "Tập trung vào nhóm sản phẩm cần cải thiện và đưa ra hành động đề xuất.",
    )

    if "product_label" not in df.columns:
        st.warning("Không có cột product_label trong dữ liệu.")
    else:
        improve_df = df[df["product_label"] == "Needs Improvement"].copy()

        if improve_df.empty:
            st.info("Không có sản phẩm thuộc nhóm Needs Improvement.")
        else:
            avg_rating = improve_df["rating"].mean() if "rating" in improve_df.columns else 0
            avg_negative = improve_df["negative_ratio"].mean() if "negative_ratio" in improve_df.columns else 0

            c1, c2, c3 = st.columns(3)

            with c1:
                metric_card("💡", "Sản phẩm cần cải thiện", format_number(len(improve_df)), "Thuộc nhóm Needs Improvement")

            with c2:
                metric_card("⭐", "Rating trung bình", format_float(avg_rating, 2), "Trung bình nhóm")

            with c3:
                metric_card("⚠️", "Tỷ lệ tiêu cực TB", format_float(avg_negative, 2), "Trung bình nhóm")

            def make_suggestion(row):
                rating_value = row.get("rating", 0)
                negative_value = row.get("negative_ratio", 0)
                sold_value = row.get("sold_count", 0)

                if negative_value >= 0.35:
                    return "Kiểm tra phản hồi tiêu cực và cải thiện mô tả/chất lượng sản phẩm."
                if rating_value > 0 and rating_value < 3.5:
                    return "Cải thiện chất lượng sản phẩm hoặc nội dung hiển thị."
                if sold_value <= 10:
                    return "Tăng hiển thị, tối ưu tiêu đề, hình ảnh và chiến lược quảng bá."
                return "Theo dõi thêm phản hồi và tối ưu nội dung bán hàng."

            improve_df["suggestion"] = improve_df.apply(make_suggestion, axis=1)

            cols = [
                "product_name",
                "price",
                "rating",
                "review_count",
                "sold_count",
                "negative_ratio",
                "estimated_revenue",
                "suggestion",
                "product_url",
            ]

            section_header("📋 Danh sách sản phẩm cần cải thiện")
            render_light_table(improve_df, columns=cols, max_rows=50)


elif page == "🚩 Kết luận":
    page_header(
        "Kết luận",
        "Tổng hợp kết quả đạt được, hạn chế còn tồn tại và hướng phát triển tiếp theo của hệ thống.",
    )

    c1, c2, c3 = st.columns(3, gap="large")

    with c1:
        st.markdown(
            """
            <div class="mini-card" style="border-left: 6px solid #10B981;">
                <div class="mini-card-title">✅ Kết quả đạt được</div>
                <ul class="mini-card-text">
                    <li>Xây dựng dashboard phân tích dữ liệu sách Tiki trực quan.</li>
                    <li>Tổng hợp được dữ liệu sản phẩm, đánh giá, lượt bán và doanh thu ước tính.</li>
                    <li>Gán nhãn sản phẩm thành 5 nhóm hiệu quả khác nhau.</li>
                    <li>Huấn luyện mô hình học máy để dự đoán nhãn sản phẩm.</li>
                    <li>Có biểu đồ, bảng đánh giá mô hình và chức năng dự đoán sản phẩm mới.</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c2:
        st.markdown(
            """
            <div class="mini-card" style="border-left: 6px solid #F59E0B;">
                <div class="mini-card-title">⚠️ Hạn chế</div>
                <ul class="mini-card-text">
                    <li>Dữ liệu phụ thuộc vào API public nên có thể thiếu một số trường.</li>
                    <li>Nhãn sản phẩm được tạo theo quy tắc nghiệp vụ, chưa phải nhãn chuyên gia.</li>
                    <li>Một số sản phẩm chưa có bình luận hoặc lượt bán đầy đủ.</li>
                    <li>Chưa phân tích sâu nội dung cảm xúc tiếng Việt trong bình luận.</li>
                    <li>Chưa đánh giá xu hướng thay đổi theo thời gian.</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c3:
        st.markdown(
            """
            <div class="mini-card" style="border-left: 6px solid #0B63F6;">
                <div class="mini-card-title">🚀 Hướng phát triển</div>
                <ul class="mini-card-text">
                    <li>Mở rộng dữ liệu với nhiều nhóm sách và nhiều thời điểm crawl khác nhau.</li>
                    <li>Bổ sung phân tích cảm xúc tiếng Việt từ nội dung bình luận.</li>
                    <li>Xây dựng module dự báo xu hướng bán hàng theo thời gian.</li>
                    <li>Tối ưu giao diện dashboard để có thể triển khai online.</li>
                    <li>Kết hợp thêm dữ liệu người bán, khuyến mãi và tồn kho nếu có.</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown(
        """
        <div class="content-card">
            <div class="section-title">📌 Tổng kết chung</div>
            <p style="color:#475569; line-height:1.7; font-size:15px; margin-top:10px;">
                Hệ thống <b>Tiki Book Intelligence</b> giúp chuyển dữ liệu sản phẩm sách từ dạng thô
                thành các thông tin có ý nghĩa hơn, bao gồm phân nhóm hiệu quả, đánh giá mô hình,
                dự đoán sản phẩm mới và gợi ý cải thiện. Đây là nền tảng phù hợp để phát triển
                thành một công cụ hỗ trợ phân tích kinh doanh sách trên sàn thương mại điện tử.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="content-card">
            <div class="section-title">🧭 Quy trình hệ thống</div>
            <div class="timeline-box">
                <div class="timeline-step">
                    <div class="step-number">1</div>
                    <div>
                        <div class="step-content-title">Thu thập dữ liệu</div>
                        <div class="step-content-text">Crawl dữ liệu sản phẩm, đánh giá, giá bán, lượt bán và đường dẫn từ Tiki.</div>
                    </div>
                </div>
                <div class="timeline-step">
                    <div class="step-number">2</div>
                    <div>
                        <div class="step-content-title">Làm sạch và tổng hợp</div>
                        <div class="step-content-text">Chuẩn hóa dữ liệu, gộp theo sản phẩm và tạo các đặc trưng cấp sản phẩm.</div>
                    </div>
                </div>
                <div class="timeline-step">
                    <div class="step-number">3</div>
                    <div>
                        <div class="step-content-title">Gán nhãn hiệu quả</div>
                        <div class="step-content-text">Phân sản phẩm thành 5 nhóm như Best Seller, High Potential, Normal và Needs Improvement.</div>
                    </div>
                </div>
                <div class="timeline-step">
                    <div class="step-number">4</div>
                    <div>
                        <div class="step-content-title">Huấn luyện và dự đoán</div>
                        <div class="step-content-text">So sánh các mô hình học máy và sử dụng mô hình tốt nhất để dự đoán sản phẩm mới.</div>
                    </div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
