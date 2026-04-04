#!/usr/bin/env python3
"""
AutoML Pipeline v2 — Production-Grade Automated Model Builder
==============================================================
sklearn Pipelines · Optuna Tuning · Interaction Features
Repeated K-Fold CV · MLflow Tracking · Structured Logging

Usage:
    python first.py <file_path> <target_column> [--test-size 0.2] [--trials 50]

Example:
    python first.py data.csv price --test-size 0.25 --trials 30
"""

# ─── 1. IMPORTS ──────────────────────────────────────────────────────────────────
import sys
import os
import argparse
import warnings
import time
import logging

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import (
    train_test_split, cross_val_score,
    RepeatedStratifiedKFold, RepeatedKFold,
)
from sklearn.preprocessing import (
    StandardScaler, RobustScaler, LabelEncoder, OneHotEncoder, PolynomialFeatures,
    FunctionTransformer, OrdinalEncoder,
)
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import (
    SelectKBest, f_classif, f_regression,
    VarianceThreshold, RFE,
)
from sklearn.linear_model import LinearRegression as _LinearReg  # for RFE base
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline as SKPipeline

# Classification models
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier,
    ExtraTreesClassifier, AdaBoostClassifier,
)
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.compose import ColumnTransformer
# Regression models
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import (
    RandomForestRegressor, GradientBoostingRegressor,
    ExtraTreesRegressor, AdaBoostRegressor,
)
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor

# Metrics
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    roc_curve, auc, f1_score, precision_recall_curve,
    average_precision_score,
    mean_absolute_error, mean_squared_error, r2_score,
)

# scipy for skewness detection
try:
    from scipy import stats as scipy_stats
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

warnings.filterwarnings("ignore")

# ─── OPTIONAL IMPORTS (graceful fallback) ─────────────────────────────────────────
try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    HAS_OPTUNA = True
except ImportError:
    HAS_OPTUNA = False

try:
    import mlflow
    import mlflow.sklearn
    HAS_MLFLOW = True
except ImportError:
    HAS_MLFLOW = False

try:
    from imblearn.over_sampling import SMOTE
    HAS_IMBLEARN = True
except ImportError:
    HAS_IMBLEARN = False

try:
    from xgboost import XGBClassifier, XGBRegressor
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

try:
    from lightgbm import LGBMClassifier, LGBMRegressor
    HAS_LGBM = True
except ImportError:
    HAS_LGBM = False

try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False


# ─── 2. LOGGING SETUP ────────────────────────────────────────────────────────────

class ColorFormatter(logging.Formatter):
    """Colored log output for terminal readability."""
    COLORS = {
        "DEBUG":    "\033[90m",
        "INFO":     "\033[94m",
        "WARNING":  "\033[93m",
        "ERROR":    "\033[91m",
        "CRITICAL": "\033[95m",
    }
    RESET = "\033[0m"
    BOLD  = "\033[1m"

    ICONS = {
        "DEBUG":    "·",
        "INFO":     "ℹ",
        "WARNING":  "⚠",
        "ERROR":    "✖",
        "CRITICAL": "‼",
    }

    def format(self, record):
        color = self.COLORS.get(record.levelname, "")
        icon = self.ICONS.get(record.levelname, "")
        msg = super().format(record)
        return f"  {color}{icon}{self.RESET}  {msg}"


def setup_logging(output_dir="eda_output"):
    """Configure logging to console (colored) + file."""
    os.makedirs(output_dir, exist_ok=True)
    logger = logging.getLogger("automl")
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()

    # Console handler — colored
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(ColorFormatter("%(message)s"))
    logger.addHandler(ch)

    # File handler — detailed
    fh = logging.FileHandler(os.path.join(output_dir, "automl.log"), mode="w")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S"))
    logger.addHandler(fh)

    return logger


log = logging.getLogger("automl")

# ─── TERMINAL FORMATTING ─────────────────────────────────────────────────────────
B = "\033[1m"
C_HEADER = "\033[95m"
C_CYAN   = "\033[96m"
C_GREEN  = "\033[92m"
END      = "\033[0m"


def banner(text, color=C_HEADER):
    w = 70
    print(f"\n{color}{B}{'═' * w}")
    print(f"  {text.upper()}")
    print(f"{'═' * w}{END}\n")


def section(text, color=C_CYAN):
    print(f"\n{color}{B}── {text} {'─' * max(0, 55 - len(text))}{END}")


# ─── 3. DATA LOADING & EDA ───────────────────────────────────────────────────────

def load_data(file_path, target_col):
    """Load dataset and validate target column."""
    banner("Loading Data")

    if not os.path.isfile(file_path):
        log.error(f"File not found: {file_path}")
        sys.exit(1)

    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".csv":
        df = pd.read_csv(file_path)
    elif ext in [".xls", ".xlsx"]:
        df = pd.read_excel(file_path)
    elif ext == ".parquet":
        df = pd.read_parquet(file_path)
    elif ext == ".json":
        df = pd.read_json(file_path)
    else:
        log.error(f"Unsupported file format: {ext}")
        sys.exit(1)

    log.info(f"Loaded {file_path}  →  {df.shape[0]} rows × {df.shape[1]} columns")

    if target_col not in df.columns:
        log.error(f"Target column '{target_col}' not found. Available: {list(df.columns)}")
        sys.exit(1)

    return df


def run_eda(df, target_col, output_dir="eda_output"):
    """Perform EDA and save plots."""
    banner("Exploratory Data Analysis")
    os.makedirs(output_dir, exist_ok=True)

    # ── Overview ──────────────────────────────────────────────────────────────
    section("Dataset Overview")
    print(f"\n{df.head(10).to_string()}\n")

    section("Data Types")
    for dtype, count in df.dtypes.value_counts().items():
        log.info(f"{dtype}: {count} columns")

    section("Statistical Summary")
    print(f"\n{df.describe(include='all').round(2).to_string()}\n")

    # ── Missing values ────────────────────────────────────────────────────────
    section("Missing Values")
    missing = df.isnull().sum()
    missing_pct = (missing / len(df) * 100).round(2)
    missing_df = pd.DataFrame({"Count": missing, "Percent": missing_pct})
    missing_df = missing_df[missing_df["Count"] > 0].sort_values("Percent", ascending=False)
    if missing_df.empty:
        log.info("No missing values found!")
    else:
        log.warning(f"{len(missing_df)} column(s) have missing values:")
        print(f"\n{missing_df.to_string()}\n")

    # ── Target distribution ───────────────────────────────────────────────────
    section("Target Column Analysis")
    n_unique = df[target_col].nunique()
    log.info(f"Unique values in '{target_col}': {n_unique}")

    if n_unique <= 20:
        log.info("Detected as CLASSIFICATION target")
        print(f"\n{df[target_col].value_counts().to_string()}\n")
        plt.figure(figsize=(8, 4))
        sns.countplot(x=target_col, data=df, palette="viridis")
        plt.title(f"Target Distribution: {target_col}")
        plt.tight_layout()
        path = os.path.join(output_dir, "target_distribution.png")
        plt.savefig(path, dpi=150); plt.close()
        log.info(f"Saved → {path}")
    else:
        log.info("Detected as REGRESSION target")
        plt.figure(figsize=(8, 4))
        sns.histplot(df[target_col].dropna(), kde=True, color="steelblue", bins=40)
        plt.title(f"Target Distribution: {target_col}")
        plt.tight_layout()
        path = os.path.join(output_dir, "target_distribution.png")
        plt.savefig(path, dpi=150); plt.close()
        log.info(f"Saved → {path}")

    # ── Correlation heatmap ───────────────────────────────────────────────────
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if len(numeric_cols) >= 2:
        section("Correlation Heatmap")
        corr = df[numeric_cols].corr()
        sz = (min(20, len(numeric_cols) * 0.8 + 2), min(16, len(numeric_cols) * 0.6 + 2))
        plt.figure(figsize=sz)
        sns.heatmap(corr, annot=len(numeric_cols) <= 15, fmt=".2f",
                    cmap="coolwarm", center=0, square=True, linewidths=0.5)
        plt.title("Feature Correlation Heatmap")
        plt.tight_layout()
        path = os.path.join(output_dir, "correlation_heatmap.png")
        plt.savefig(path, dpi=150); plt.close()
        log.info(f"Saved → {path}")

    # ── Feature distributions ─────────────────────────────────────────────────
    section("Feature Distributions")
    plot_cols = [c for c in numeric_cols if c != target_col][:12]
    if plot_cols:
        n = len(plot_cols)
        ncols = min(4, n)
        nrows = (n + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows))
        axes = np.array(axes).flatten() if n > 1 else [axes]
        for i, col in enumerate(plot_cols):
            sns.histplot(df[col].dropna(), kde=True, ax=axes[i], color="steelblue", bins=30)
            axes[i].set_title(col, fontsize=10); axes[i].set_xlabel("")
        for j in range(i + 1, len(axes)):
            axes[j].set_visible(False)
        plt.suptitle("Numeric Feature Distributions", fontsize=13, y=1.02)
        plt.tight_layout()
        path = os.path.join(output_dir, "feature_distributions.png")
        plt.savefig(path, dpi=150, bbox_inches="tight"); plt.close()
        log.info(f"Saved → {path}")

    log.info(f"All EDA plots saved to: {os.path.abspath(output_dir)}/")
    return df


# ─── 3.5 DATA CLEANING ───────────────────────────────────────────────────────────

def drop_useless_columns(df, target_col):
    """
    Drop columns that provide no predictive signal:
    - Constant columns (only 1 unique value)
    - Near-constant columns (>99.5% same value)
    - ID-like columns (all unique values AND not the target)
    """
    section("Useless Column Detection")
    to_drop = []

    for col in df.columns:
        if col == target_col:
            continue

        n_unique = df[col].nunique(dropna=True)
        n_rows = len(df)

        # Constant column
        if n_unique <= 1:
            to_drop.append((col, "constant (0-1 unique values)"))
            continue

        # Near-constant (>99.5% same value)
        top_freq = df[col].value_counts(normalize=True, dropna=False).iloc[0]
        if top_freq >= 0.995:
            to_drop.append((col, f"near-constant (top value = {top_freq:.1%} of rows)"))
            continue

        # ID-like: all unique AND non-numeric or integer column
        if n_unique == n_rows:
            if pd.api.types.is_integer_dtype(df[col]) or df[col].dtype == object:
                to_drop.append((col, f"ID-like (all {n_unique} values unique)"))

    if to_drop:
        log.warning(f"Dropping {len(to_drop)} useless column(s):")
        for col, reason in to_drop:
            log.warning(f"  {col!r}: {reason}")
        df = df.drop(columns=[c for c, _ in to_drop])
    else:
        log.info("No useless columns detected")

    return df


def clean_data(df, target_col, clip_outliers=True, output_dir="eda_output"):
    """
    Data cleaning step — runs BEFORE EDA and prepare_data.
    Handles: duplicates, outliers (IQR Winsorization), inconsistent categoricals.
    """
    banner("Data Cleaning")
    df = df.copy()
    original_shape = df.shape

    # ── 1. Duplicate removal ──────────────────────────────────────────────────
    section("Duplicate Removal")
    n_dups = df.duplicated().sum()
    if n_dups > 0:
        df = df.drop_duplicates().reset_index(drop=True)
        log.warning(f"Removed {n_dups} duplicate rows → {df.shape[0]} rows remain")
    else:
        log.info("No duplicate rows found")

    # ── 1.5 Drop useless columns ──────────────────────────────────────────────
    df = drop_useless_columns(df, target_col)

    # ── 2. Inconsistent categoricals — strip & lowercase ─────────────────────
    section("Categorical Normalization")
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    if target_col in cat_cols:
        cat_cols.remove(target_col)
    normalized = []
    for col in cat_cols:
        before = df[col].nunique()
        df[col] = df[col].astype(str).str.strip().str.lower()
        after = df[col].nunique()
        if after < before:
            normalized.append(col)
    if normalized:
        log.info(f"Normalized {len(normalized)} column(s) (strip/lower reduced cardinality): {normalized}")
    else:
        log.info("No cardinality reduction from normalization")

    # ── 3. Outlier detection & Winsorization (IQR) ───────────────────────────
    section("Outlier Detection (IQR)")
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if target_col in numeric_cols:
        numeric_cols.remove(target_col)

    outlier_report = []
    for col in numeric_cols:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        n_out = ((df[col] < lower) | (df[col] > upper)).sum()
        pct = n_out / len(df) * 100
        if n_out > 0:
            outlier_report.append((col, n_out, round(pct, 2)))

    if outlier_report:
        log.warning(f"Outliers found in {len(outlier_report)} feature(s):")
        for col, n, pct in outlier_report[:10]:
            log.warning(f"  {col}: {n} rows ({pct}%)")
        if clip_outliers:
            log.info("Applying Winsorization (clip to 1st–99th percentile) on feature columns")
            for col in numeric_cols:
                lo = df[col].quantile(0.01)
                hi = df[col].quantile(0.99)
                df[col] = df[col].clip(lower=lo, upper=hi)
        else:
            log.info("clip_outliers=False — outliers reported but NOT removed")
    else:
        log.info("No significant outliers detected")

    log.info(f"Cleaning complete: {original_shape} → {df.shape}")
    return df


# ─── 3.7 DATETIME FEATURE EXTRACTION ────────────────────────────────────────────

def extract_datetime_features(X):
    """
    Detect datetime-parseable columns, extract temporal features,
    and drop the original datetime column.
    Returns (X_new, list_of_added_cols).
    """
    added = []
    dt_cols = []

    for col in X.columns:
        # Skip if already numeric
        if pd.api.types.is_numeric_dtype(X[col]):
            continue
        # Try parsing as datetime
        try:
            parsed = pd.to_datetime(X[col], infer_datetime_format=True, errors="raise")
            dt_cols.append((col, parsed))
        except Exception:
            pass  # not a datetime column, skip

    for col, parsed in dt_cols:
        log.info(f"Detected datetime column: '{col}' — extracting temporal features")
        X = X.copy()
        X[f"{col}_year"]       = parsed.dt.year
        X[f"{col}_month"]      = parsed.dt.month       # 1-12
        X[f"{col}_day"]        = parsed.dt.day         # 1-31
        X[f"{col}_dayofweek"]  = parsed.dt.dayofweek   # 0=Monday, 6=Sunday
        X[f"{col}_hour"]       = parsed.dt.hour        # 0-23 (0 if no time)
        X[f"{col}_is_weekend"] = parsed.dt.dayofweek.isin([5, 6]).astype(int)
        X[f"{col}_quarter"]    = parsed.dt.quarter     # 1-4

        added += [
            f"{col}_year", f"{col}_month", f"{col}_day",
            f"{col}_dayofweek", f"{col}_hour",
            f"{col}_is_weekend", f"{col}_quarter"
        ]
        X = X.drop(columns=[col])  # drop original datetime string

    return X, added




def prepare_data(df, target_col):
    """Separate X/y, encode target if needed. NO scaling here (done inside Pipeline)."""
    banner("Preparing Data")

    X = df.drop(columns=[target_col])
    y = df[target_col].copy()

    # ── Drop rows where target is NaN ─────────────────────────────────────────
    if y.isnull().sum() > 0:
        mask = y.notna()
        log.warning(f"Dropping {(~mask).sum()} rows where target is NaN")
        X = X[mask].reset_index(drop=True)
        y = y[mask].reset_index(drop=True)

    # ── Encode target if categorical ──────────────────────────────────────────
    target_le = None
    if y.dtype == "object" or y.dtype.name == "category":
        section("Encoding Target Variable")
        target_le = LabelEncoder()
        y = pd.Series(target_le.fit_transform(y), name=target_col)
        mapping = dict(zip(target_le.classes_, target_le.transform(target_le.classes_)))
        log.info(f"Label-encoded '{target_col}' → {mapping}")

    # ── Extract datetime features ──────────────────────────────────────────────
    X, dt_added = extract_datetime_features(X)
    if dt_added:
        log.info(f"Added {len(dt_added)} datetime-derived features, dropped original datetime col(s)")

    # ── Identify column types ─────────────────────────────────────────────────
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    all_cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()

    # Split categoricals: low-cardinality (OHE) vs high-cardinality (Ordinal)
    CARDINALITY_THRESHOLD = 10
    low_card_cols  = [c for c in all_cat_cols if X[c].nunique() <= CARDINALITY_THRESHOLD]
    high_card_cols = [c for c in all_cat_cols if X[c].nunique() >  CARDINALITY_THRESHOLD]

    # ── Detect free-text columns (high-cardinality + long avg string length) ──
    TEXT_AVG_LENGTH_THRESHOLD = 40  # avg chars above this → treat as free text
    text_cols = []
    filtered_high_card = []
    for col in high_card_cols:
        avg_len = X[col].dropna().astype(str).str.len().mean()
        if avg_len > TEXT_AVG_LENGTH_THRESHOLD:
            text_cols.append(col)
            log.warning(f"Column '{col}' detected as free-text (avg length={avg_len:.0f} chars) → TF-IDF")
        else:
            filtered_high_card.append(col)
    high_card_cols = filtered_high_card

    # Detect skewed numeric columns (skewness > 1.0) for log1p transform
    skewed_cols = []
    if HAS_SCIPY:
        for col in numeric_cols:
            vals = X[col].dropna()
            if len(vals) > 10 and vals.min() >= 0:   # log1p requires non-negative
                skew = float(scipy_stats.skew(vals))
                if abs(skew) > 1.0:
                    skewed_cols.append(col)
        if skewed_cols:
            log.info(f"Skewed features (|skew|>1, will apply log1p): {skewed_cols}")

    section("Column Type Detection")
    log.info(f"Numeric cols ({len(numeric_cols)}): {numeric_cols[:10]}{'...' if len(numeric_cols) > 10 else ''}")
    log.info(f"Low-card cat cols  ({len(low_card_cols)},  ≤{CARDINALITY_THRESHOLD} unique) → OHE: {low_card_cols[:5]}")
    log.info(f"High-card cat cols ({len(high_card_cols)}, >{CARDINALITY_THRESHOLD} unique) → Ordinal: {high_card_cols[:5]}")
    if text_cols:
        log.info(f"Free-text cols ({len(text_cols)}) → TF-IDF: {text_cols[:5]}")
    if skewed_cols:
        log.info(f"Skewed numeric cols → log1p: {skewed_cols[:10]}")

    return X, y, numeric_cols, low_card_cols, high_card_cols, skewed_cols, target_le, text_cols


def build_preprocessor(numeric_cols, low_card_cols, high_card_cols,
                        skewed_cols=None, add_interactions=False, text_cols=None):
    """
    Build a ColumnTransformer:
    - Numeric: impute → (log1p for skewed) → interact → scale
    - Low-cardinality cats: impute → OHE
    - High-cardinality cats: impute → OrdinalEncoder
    - Free-text cols: impute → TF-IDF
    """
    skewed_cols = skewed_cols or []
    text_cols = text_cols or []

    # ── Numeric pipeline ──────────────────────────────────────────────────────
    non_skewed = [c for c in numeric_cols if c not in skewed_cols]
    transformers = []

    if skewed_cols:
        skewed_steps = [
            ("imputer", SimpleImputer(strategy="median")),
            ("log1p", FunctionTransformer(np.log1p, validate=True)),
            ("scaler", RobustScaler()),
        ]
        transformers.append(("num_skewed", SKPipeline(skewed_steps), skewed_cols))
        log.info(f"log1p pipeline for {len(skewed_cols)} skewed column(s)")

    if non_skewed:
        numeric_steps = [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", RobustScaler()),
        ]
        if add_interactions and (len(non_skewed) + len(skewed_cols)) <= 50 \
                and len(numeric_cols) >= 2:
            numeric_steps.insert(1, (
                "interactions",
                PolynomialFeatures(degree=2, interaction_only=True, include_bias=False),
            ))
            log.info(f"PolynomialFeatures(interaction_only=True) for {len(non_skewed)} non-skewed cols")
        transformers.append(("num", SKPipeline(numeric_steps), non_skewed))
    elif not skewed_cols:
        # fallback: all numeric go through standard pipeline
        numeric_steps = [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", RobustScaler()),
        ]
        transformers.append(("num", SKPipeline(numeric_steps), numeric_cols))

    # ── Low-cardinality categorical → OHE ────────────────────────────────────
    if low_card_cols:
        ohe_pipeline = SKPipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False, drop="first")),
        ])
        transformers.append(("cat_low", ohe_pipeline, low_card_cols))

    # ── High-cardinality categorical → OrdinalEncoder ─────────────────────────
    if high_card_cols:
        ord_pipeline = SKPipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OrdinalEncoder(
                handle_unknown="use_encoded_value", unknown_value=-1,
            )),
        ])
        transformers.append(("cat_high", ord_pipeline, high_card_cols))
        log.info(f"OrdinalEncoder for {len(high_card_cols)} high-cardinality col(s)")

    # ── Free-text columns → TF-IDF ────────────────────────────────────────────
    if text_cols:
        for col in text_cols:
            tfidf_pipeline = SKPipeline([
                ("imputer", SimpleImputer(strategy="constant", fill_value="")),
                ("flatten", FunctionTransformer(lambda x: x.ravel(), validate=False)),
                ("tfidf", TfidfVectorizer(max_features=100, stop_words="english")),
            ])
            transformers.append((f"text_{col}", tfidf_pipeline, [col]))
        log.info(f"TF-IDF pipelines added for {len(text_cols)} text column(s) (max_features=100)")

    preprocessor = ColumnTransformer(transformers=transformers, remainder="drop")
    log.info("Built ColumnTransformer (clean → log1p → interact → scale → encode)")
    return preprocessor


# ─── 5. FEATURE SELECTION ────────────────────────────────────────────────────────

class _FeatureSelectionPipeline:
    """
    3-stage feature selector (wraps multiple filters so test data transforms identically):
      Stage 1 — VarianceThreshold  (remove near-constant features)
      Stage 2 — Correlation drop   (remove one from each highly-correlated pair)
      Stage 3 — SelectKBest        (keep top k by ANOVA-F / F-regression)
    Exposes .transform() for both train and test data.
    """
    def __init__(self, var_selector, corr_mask, kbest_selector):
        self.var_selector   = var_selector
        self.corr_mask      = corr_mask      # boolean array after variance step
        self.kbest_selector = kbest_selector

    def transform(self, X):
        X = self.var_selector.transform(X)
        X = X[:, self.corr_mask]
        if self.kbest_selector is not None:
            X = self.kbest_selector.transform(X)
        return X


def select_features_from_transformed(X_train_t, y_train, feature_names, is_classification, output_dir="eda_output"):
    """
    3-stage feature selection on already-transformed data.
    Returns a _FeatureSelectionPipeline for consistent train/test transform.
    """
    banner("Feature Selection")
    n_orig = X_train_t.shape[1]
    log.info(f"Features after preprocessing: {n_orig}")

    if n_orig <= 5:
        log.info("Only a few features — skipping selection")
        return None

    # ── Stage 1: VarianceThreshold ────────────────────────────────────────────
    section("Stage 1 — Variance Threshold")
    var_sel = VarianceThreshold(threshold=0.01)
    X_v = var_sel.fit_transform(X_train_t)
    curr_features = feature_names[var_sel.get_support()] if feature_names is not None else None
    
    n_after_var = X_v.shape[1]
    log.info(f"VarianceThreshold: {n_orig} → {n_after_var} features "
             f"(dropped {n_orig - n_after_var} near-constant)")

    # ── Stage 2: Correlation drop ─────────────────────────────────────────────
    section("Stage 2 — Correlation Drop (threshold=0.95)")
    corr_matrix = np.abs(np.corrcoef(X_v.T))
    upper = np.triu(corr_matrix, k=1)
    to_drop = set()
    for i in range(upper.shape[1]):
        if any(upper[:, i] > 0.95):
            to_drop.add(i)
    corr_mask = np.array([i not in to_drop for i in range(X_v.shape[1])])
    X_c = X_v[:, corr_mask]
    if curr_features is not None:
        curr_features = curr_features[corr_mask]
        
    n_after_corr = X_c.shape[1]
    log.info(f"Correlation drop: {n_after_var} → {n_after_corr} features "
             f"(dropped {n_after_var - n_after_corr} highly-correlated)")

    # ── Stage 3: SelectKBest ──────────────────────────────────────────────────
    section("Stage 3 — SelectKBest")
    kbest_selector = None
    if n_after_corr > 5:
        k = min(n_after_corr, max(10, n_after_corr * 2 // 3))
        score_func = f_classif if is_classification else f_regression
        kbest_selector = SelectKBest(score_func=score_func, k=k)
        kbest_selector.fit(X_c, y_train)
        n_final = kbest_selector.get_support().sum()
        log.info(f"SelectKBest: {n_after_corr} → {n_final} features (k={k})")

        scores = kbest_selector.scores_[kbest_selector.get_support()]
        if curr_features is not None:
             final_names = curr_features[kbest_selector.get_support()]
        else:
             final_names = [f"F{i}" for i in range(len(scores))]

        fi_df = pd.DataFrame({
            "Feature": final_names,
            "Score": scores,
        }).sort_values("Score", ascending=True)
        plt.figure(figsize=(8, max(3, len(fi_df) * 0.25)))
        plt.barh(fi_df["Feature"].astype(str), fi_df["Score"], color="steelblue")
        plt.title("Feature Importance (SelectKBest Scores)")
        plt.xlabel("Score"); plt.tight_layout()
        path = os.path.join(output_dir, "feature_importance.png")
        plt.savefig(path, dpi=150); plt.close()
        log.info(f"Saved → {path}")
    else:
        log.info(f"Only {n_after_corr} features after correlation drop — skipping SelectKBest")

    n_final_total = n_after_corr if kbest_selector is None else int(kbest_selector.get_support().sum())
    log.info(f"Feature selection complete: {n_orig} → {n_final_total} features")
    return _FeatureSelectionPipeline(var_sel, corr_mask, kbest_selector)


# ─── 5.5 CLASS IMBALANCE HANDLING ────────────────────────────────────────────────

def handle_class_imbalance(X_train, y_train, is_classification):
    """
    Detect and handle class imbalance (classification only).
    Uses SMOTE if imbalanced-learn is installed, else logs a warning.
    Returns (X_res, y_res, imbalanced: bool).
    """
    if not is_classification:
        return X_train, y_train, False

    banner("Class Imbalance Check")
    classes, counts = np.unique(y_train, return_counts=True)
    total = len(y_train)
    minority_ratio = counts.min() / total

    section("Class Distribution")
    for cls, cnt in zip(classes, counts):
        bar = "█" * int(cnt / total * 40)
        print(f"  Class {cls}: {cnt:>5} ({cnt/total*100:.1f}%)  {bar}")

    IMBALANCE_THRESHOLD = 0.20
    if minority_ratio >= IMBALANCE_THRESHOLD:
        log.info(f"Classes balanced (minority={minority_ratio:.2%}) — no resampling")
        return X_train, y_train, False

    log.warning(f"Imbalance detected! minority ratio={minority_ratio:.2%}")

    if HAS_IMBLEARN:
        section("Applying SMOTE")
        try:
            smote = SMOTE(random_state=42)
            X_res, y_res = smote.fit_resample(X_train, y_train)
            new_counts = np.unique(y_res, return_counts=True)[1]
            log.info(f"SMOTE: {total} → {len(y_res)} samples. Counts: {dict(zip(classes, new_counts))}")
            return X_res, y_res, True
        except Exception as e:
            log.warning(f"SMOTE failed ({e}) — proceeding without resampling")
    else:
        log.warning("imbalanced-learn not installed → pip install imbalanced-learn")
        log.warning("Falling back to class_weight='balanced' on supported models")

    return X_train, y_train, True  # imbalanced=True triggers class_weight injection


# ─── 6. OPTUNA HYPERPARAMETER TUNING ─────────────────────────────────────────────

def _get_optuna_search_space(trial, model_name, is_classification):
    """Define per-model Optuna search space."""
    if model_name == "Logistic Regression":
        return {
            "C": trial.suggest_float("C", 1e-3, 100, log=True),
            "max_iter": 2000,
            "random_state": 42,
        }
    elif model_name == "Random Forest":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 50, 300, step=50),
            "max_depth": trial.suggest_int("max_depth", 3, 20),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
            "random_state": 42,
            "n_jobs": -1,
        }
    elif model_name == "Gradient Boosting":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 50, 300, step=50),
            "max_depth": trial.suggest_int("max_depth", 2, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "random_state": 42,
        }
    elif model_name in ("SVM", "SVR"):
        params = {
            "C": trial.suggest_float("C", 1e-2, 100, log=True),
            "kernel": trial.suggest_categorical("kernel", ["rbf", "linear", "poly"]),
            "gamma": trial.suggest_categorical("gamma", ["scale", "auto"]),
        }
        if is_classification:
            params["probability"] = True
            params["random_state"] = 42
        return params
    elif model_name == "KNN":
        return {
            "n_neighbors": trial.suggest_int("n_neighbors", 3, 15, step=2),
            "weights": trial.suggest_categorical("weights", ["uniform", "distance"]),
        }
    elif model_name == "Decision Tree":
        return {
            "max_depth": trial.suggest_int("max_depth", 2, 20),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
            "criterion": trial.suggest_categorical(
                "criterion",
                ["gini", "entropy"] if is_classification else ["squared_error", "friedman_mse"],
            ),
            "random_state": 42,
        }
    elif model_name == "Extra Trees":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 50, 300, step=50),
            "max_depth": trial.suggest_int("max_depth", 3, 20),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
            "random_state": 42,
            "n_jobs": -1,
        }
    elif model_name == "AdaBoost":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 50, 300, step=50),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 2.0, log=True),
            "random_state": 42,
        }
    elif model_name == "XGBoost":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 50, 300, step=50),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "random_state": 42,
            "n_jobs": -1,
        }
    elif model_name == "LightGBM":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 50, 300, step=50),
            "max_depth": trial.suggest_int("max_depth", 3, 15),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 20, 100),
            "random_state": 42,
            "n_jobs": -1,
            "verbose": -1,
        }
    elif model_name == "Ridge":
        return {"alpha": trial.suggest_float("alpha", 1e-3, 100, log=True)}
    elif model_name == "Lasso":
        return {
            "alpha": trial.suggest_float("alpha", 1e-4, 10, log=True),
            "max_iter": 5000,
        }
    elif model_name in ("Linear Regression", "Naive Bayes"):
        return {}  # no tunable hyperparams
    return {}


def _make_estimator(model_name, params, is_classification):
    """Instantiate an estimator given name + params."""
    if is_classification:
        cls_map = {
            "Logistic Regression": LogisticRegression,
            "Random Forest":       RandomForestClassifier,
            "Gradient Boosting":   GradientBoostingClassifier,
            "Decision Tree":       DecisionTreeClassifier,
            "Extra Trees":         ExtraTreesClassifier,
            "AdaBoost":            AdaBoostClassifier,
            "SVM":                 SVC,
            "KNN":                 KNeighborsClassifier,
            "Naive Bayes":         GaussianNB,
        }
        if HAS_XGB: cls_map["XGBoost"] = XGBClassifier
        if HAS_LGBM: cls_map["LightGBM"] = LGBMClassifier
    else:
        cls_map = {
            "Linear Regression":  LinearRegression,
            "Ridge":              Ridge,
            "Lasso":              Lasso,
            "Random Forest":      RandomForestRegressor,
            "Gradient Boosting":  GradientBoostingRegressor,
            "Decision Tree":      DecisionTreeRegressor,
            "Extra Trees":        ExtraTreesRegressor,
            "AdaBoost":           AdaBoostRegressor,
            "SVR":                SVR,
            "KNN":                KNeighborsRegressor,
        }
        if HAS_XGB: cls_map["XGBoost"] = XGBRegressor
        if HAS_LGBM: cls_map["LightGBM"] = LGBMRegressor
    return cls_map[model_name](**params)


def tune_model_optuna(model_name, X_train, y_train, is_classification, n_trials=50, scoring=None):
    """
    Use Optuna TPE to find the best hyperparameters for a model.
    Returns (best_params, best_score).
    """
    if scoring is None:
        # Default: same metrics used in the final CV phase
        scoring = "f1_weighted" if is_classification else "neg_root_mean_squared_error"

    if is_classification:
        cv = RepeatedStratifiedKFold(n_splits=3, n_repeats=1, random_state=42)
    else:
        cv = RepeatedKFold(n_splits=3, n_repeats=1, random_state=42)

    def objective(trial):
        params = _get_optuna_search_space(trial, model_name, is_classification)
        estimator = _make_estimator(model_name, params, is_classification)
        scores = cross_val_score(estimator, X_train, y_train, cv=cv, scoring=scoring, n_jobs=-1)
        return scores.mean()

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    return study.best_params, study.best_value


# ─── 7. MODEL TRAINING ───────────────────────────────────────────────────────────

def get_default_models(is_classification):
    """Default (untuned) models — expanded model zoo."""
    if is_classification:
        models = {
            "Logistic Regression": LogisticRegression(max_iter=2000, random_state=42),
            "Decision Tree":       DecisionTreeClassifier(max_depth=10, random_state=42),
            "Random Forest":       RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
            "Extra Trees":         ExtraTreesClassifier(n_estimators=100, random_state=42, n_jobs=-1),
            "Gradient Boosting":   GradientBoostingClassifier(n_estimators=100, random_state=42),
            "AdaBoost":            AdaBoostClassifier(n_estimators=100, random_state=42),
            "SVM":                 SVC(kernel="rbf", probability=True, random_state=42),
            "KNN":                 KNeighborsClassifier(n_neighbors=5),
            "Naive Bayes":         GaussianNB(),
        }
        if HAS_XGB:
            models["XGBoost"] = XGBClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        if HAS_LGBM:
            models["LightGBM"] = LGBMClassifier(n_estimators=100, random_state=42, n_jobs=-1, verbose=-1)
        return models
    else:
        models = {
            "Linear Regression":  LinearRegression(),
            "Ridge":              Ridge(alpha=1.0),
            "Lasso":              Lasso(alpha=0.01, max_iter=5000),
            "Decision Tree":      DecisionTreeRegressor(max_depth=10, random_state=42),
            "Random Forest":      RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
            "Extra Trees":        ExtraTreesRegressor(n_estimators=100, random_state=42, n_jobs=-1),
            "Gradient Boosting":  GradientBoostingRegressor(n_estimators=100, random_state=42),
            "AdaBoost":           AdaBoostRegressor(n_estimators=100, random_state=42),
            "SVR":                SVR(kernel="rbf"),
            "KNN":                KNeighborsRegressor(n_neighbors=5),
        }
        if HAS_XGB:
            models["XGBoost"] = XGBRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        if HAS_LGBM:
            models["LightGBM"] = LGBMRegressor(n_estimators=100, random_state=42, n_jobs=-1, verbose=-1)
        return models


def train_models(X_train, X_test, y_train, y_test, is_classification,
                 n_trials=50, output_dir="eda_output", imbalanced=False):
    """
    Train all models with Repeated K-Fold CV.
    Optionally tunes via Optuna, injects class_weight when imbalanced.
    Returns (best_model, best_name, results_df, best_cv_score).
    """
    banner("Model Training")

    log.info(f"Train set: {X_train.shape[0]} samples, {X_train.shape[1]} features")
    log.info(f"Test set:  {X_test.shape[0]} samples")
    if imbalanced:
        log.warning("Imbalanced dataset — class_weight='balanced' injected into supporting models")

    # ── CV strategy (Repeated K-Fold) ─────────────────────────────────────────
    if is_classification:
        cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=42)
    else:
        cv = RepeatedKFold(n_splits=5, n_repeats=3, random_state=42)

    scoring = "f1_weighted" if is_classification else "r2"
    model_names = list(get_default_models(is_classification).keys())

    # Models that support class_weight param
    CW_MODELS = {"Logistic Regression", "SVM", "Decision Tree",
                 "Random Forest", "Extra Trees", "Gradient Boosting"}

    results = []
    trained_models = {}

    # ── Phase 1: Optuna tuning (if available) ─────────────────────────────────
    if HAS_OPTUNA and n_trials > 0:
        section(f"Bayesian Hyperparameter Tuning (Optuna, {n_trials} trials/model)")
        print()
        for name in model_names:
            t0 = time.time()
            best_params, best_val = tune_model_optuna(
                name, X_train, y_train, is_classification, n_trials=n_trials,
                scoring="f1_weighted" if is_classification else "neg_root_mean_squared_error"
            )
            elapsed = time.time() - t0
            log.info(f"{name}: best_score={best_val:.4f}, params={best_params} ({elapsed:.1f}s)")
            if imbalanced and name in CW_MODELS:
                best_params["class_weight"] = "balanced"
            model = _make_estimator(name, best_params, is_classification)
            trained_models[name] = model

            if HAS_MLFLOW:
                with mlflow.start_run(run_name=f"tune_{name}", nested=True):
                    mlflow.log_params(best_params)
                    mlflow.log_metric("optuna_best_score", best_val)
    else:
        if not HAS_OPTUNA:
            log.warning("Optuna not installed — using default hyperparameters. "
                        "Install with: pip install optuna")
        elif n_trials == 0:
            log.info("--trials 0 passed — skipping Optuna tuning, using default hyperparameters")
        defaults = get_default_models(is_classification)
        for name, model in defaults.items():
            if imbalanced and name in CW_MODELS:
                model.set_params(class_weight="balanced")
            trained_models[name] = model

    # ── Phase 2: Repeated K-Fold Cross-Validation ─────────────────────────────
    section(f"Repeated K-Fold CV (5-fold × 3 repeats, scoring={scoring})")
    print()

    for name, model in trained_models.items():
        t0 = time.time()
        cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring=scoring, n_jobs=-1)
        elapsed = time.time() - t0
        mean_s, std_s = cv_scores.mean(), cv_scores.std()
        results.append({
            "Model": name,
            "CV Mean": round(mean_s, 4),
            "CV Std": round(std_s, 4),
            "Time (s)": round(elapsed, 2),
        })
        bar = "█" * int(max(0, mean_s) * 30)
        print(f"  {name:<22s}  {mean_s:.4f} ± {std_s:.4f}  {C_GREEN}{bar}{END}  ({elapsed:.1f}s)")

        if HAS_MLFLOW:
            with mlflow.start_run(run_name=f"cv_{name}", nested=True):
                mlflow.log_metric("cv_mean", mean_s)
                mlflow.log_metric("cv_std", std_s)

    results_df = pd.DataFrame(results).sort_values("CV Mean", ascending=False)
    print(f"\n{B}{'─' * 60}{END}")
    print(results_df.to_string(index=False))
    print(f"{B}{'─' * 60}{END}\n")

    # ── Train best model on full training set ─────────────────────────────────
    best_name = results_df.iloc[0]["Model"]
    best_cv_score = results_df.iloc[0]["CV Mean"]
    best_model = trained_models[best_name]
    best_model.fit(X_train, y_train)
    log.info(f"Best model: {B}{best_name}{END} (CV: {best_cv_score} ± {results_df.iloc[0]['CV Std']})")

    return best_model, best_name, results_df, best_cv_score


# ─── 8. EVALUATION ───────────────────────────────────────────────────────────────

def evaluate(model, model_name, X_test, y_test, is_classification, target_le,
             best_cv_score=None, output_dir="eda_output"):
    """Evaluate the best model: enriched metrics + overfitting check."""
    banner("Model Evaluation")
    y_pred = model.predict(X_test)

    if is_classification:
        acc  = accuracy_score(y_test, y_pred)
        f1   = f1_score(y_test, y_pred, average="weighted")
        section(f"Classification Results — {model_name}")
        print(f"\n  {B}Accuracy  : {acc:.4f}{END}")
        print(f"  {B}F1 (w-avg): {f1:.4f}{END}\n")

        target_names = [str(c) for c in target_le.classes_] if target_le else None
        print(classification_report(y_test, y_pred, target_names=target_names))
        log.info(f"Test Accuracy={acc:.4f}  F1={f1:.4f}")

        # Overfitting check
        if best_cv_score is not None:
            gap = best_cv_score - f1
            if gap > 0.10:
                log.warning(f"Overfitting detected! CV F1={best_cv_score:.4f} vs Test F1={f1:.4f} (gap={gap:.4f})")
            else:
                log.info(f"Overfitting check OK: CV F1={best_cv_score:.4f} vs Test F1={f1:.4f} (gap={gap:.4f})")

        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=target_names, yticklabels=target_names)
        plt.title(f"Confusion Matrix — {model_name}")
        plt.ylabel("Actual"); plt.xlabel("Predicted"); plt.tight_layout()
        path = os.path.join(output_dir, "confusion_matrix.png")
        plt.savefig(path, dpi=150); plt.close()
        log.info(f"Saved → {path}")

        # ROC + Precision-Recall curves
        from sklearn.preprocessing import label_binarize
        from sklearn.metrics import roc_auc_score
        
        n_classes = len(np.unique(y_test))
        roc_auc = None
        if hasattr(model, "predict_proba"):
            y_proba = model.predict_proba(X_test)
            if n_classes == 2:
                y_proba_1 = y_proba[:, 1]
                # ROC
                fpr, tpr, thresholds_roc = roc_curve(y_test, y_proba_1)
                roc_auc = auc(fpr, tpr)
                plt.figure(figsize=(6, 5))
                plt.plot(fpr, tpr, color="steelblue", lw=2, label=f"ROC AUC = {roc_auc:.4f}")
                plt.plot([0, 1], [0, 1], "k--", lw=1)
                plt.xlabel("FPR"); plt.ylabel("TPR")
                plt.title(f"ROC Curve — {model_name}")
                plt.legend(loc="lower right"); plt.tight_layout()
                path = os.path.join(output_dir, "roc_curve.png")
                plt.savefig(path, dpi=150); plt.close()
                log.info(f"ROC AUC: {roc_auc:.4f}  |  Saved → {path}")
    
                # Precision-Recall
                pr_auc = average_precision_score(y_test, y_proba_1)
                prec, rec, _ = precision_recall_curve(y_test, y_proba_1)
                plt.figure(figsize=(6, 5))
                plt.plot(rec, prec, color="darkorange", lw=2, label=f"PR AUC = {pr_auc:.4f}")
                plt.xlabel("Recall"); plt.ylabel("Precision")
                plt.title(f"Precision-Recall Curve — {model_name}")
                plt.legend(loc="upper right"); plt.tight_layout()
                path = os.path.join(output_dir, "pr_curve.png")
                plt.savefig(path, dpi=150); plt.close()
                log.info(f"PR AUC: {pr_auc:.4f}  |  Saved → {path}")
    
                # Optimal threshold by F1
                f1s = [f1_score(y_test, (y_proba_1 >= t).astype(int)) for t in thresholds_roc]
                best_thresh = thresholds_roc[int(np.argmax(f1s))]
                log.info(f"Optimal decision threshold (max F1): {best_thresh:.4f}")
            elif n_classes > 2:
                # Multi-class OvR
                y_test_bin = label_binarize(y_test, classes=np.unique(y_test))
                roc_auc = roc_auc_score(y_test_bin, y_proba, multi_class="ovr")
                
                plt.figure(figsize=(6, 5))
                for i in range(n_classes):
                    fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_proba[:, i])
                    label = target_names[i] if target_names else f"Class {i}"
                    plt.plot(fpr, tpr, lw=2, label=f"{label} ROC")
                plt.plot([0, 1], [0, 1], "k--", lw=1)
                plt.xlabel("FPR"); plt.ylabel("TPR")
                plt.title(f"ROC Curve (Multi-class OvR) — {model_name}")
                plt.legend(loc="lower right", fontsize=8); plt.tight_layout()
                path = os.path.join(output_dir, "roc_curve_ovr.png")
                plt.savefig(path, dpi=150); plt.close()
                log.info(f"Multi-class ROC AUC: {roc_auc:.4f}  |  Saved → {path}")

        if HAS_MLFLOW:
            mlflow.log_metric("test_accuracy", acc)
            mlflow.log_metric("test_f1", f1)
            if roc_auc:
                mlflow.log_metric("test_roc_auc", roc_auc)

        if HAS_SHAP:
            fn = [f"F{i}" for i in range(X_test.shape[1])]
            explain_with_shap(model, model_name, X_test, fn, output_dir)

        return {"accuracy": acc, "f1": f1, "roc_auc": roc_auc}

    else:
        mae  = mean_absolute_error(y_test, y_pred)
        mse  = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2   = r2_score(y_test, y_pred)
        # MAPE (guard against zero targets)
        nonzero = y_test != 0
        mape = float(np.mean(np.abs((y_test[nonzero] - y_pred[nonzero]) / y_test[nonzero])) * 100) \
               if nonzero.sum() > 0 else float("nan")

        section(f"Regression Results — {model_name}")
        print(f"\n  {B}MAE  : {mae:.4f}{END}")
        print(f"  {B}MSE  : {mse:.4f}{END}")
        print(f"  {B}RMSE : {rmse:.4f}{END}")
        print(f"  {B}R²   : {r2:.4f}{END}")
        print(f"  {B}MAPE : {mape:.2f}%{END}\n")
        log.info(f"Test MAE={mae:.4f} RMSE={rmse:.4f} R²={r2:.4f} MAPE={mape:.2f}%")

        # Overfitting check (CV R2 vs test R2)
        if best_cv_score is not None:
            gap = best_cv_score - r2
            if gap > 0.10:
                log.warning(f"Overfitting: CV R²={best_cv_score:.4f} vs Test R²={r2:.4f} (gap={gap:.4f})")
            else:
                log.info(f"Overfitting check OK: CV R²={best_cv_score:.4f} vs Test R²={r2:.4f}")

        # Actual vs Predicted
        plt.figure(figsize=(6, 5))
        plt.scatter(y_test, y_pred, alpha=0.5, color="steelblue", edgecolors="white", linewidths=0.3)
        mn, mx = min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())
        plt.plot([mn, mx], [mn, mx], "r--", lw=1.5)
        plt.xlabel("Actual"); plt.ylabel("Predicted")
        plt.title(f"Actual vs Predicted — {model_name}"); plt.tight_layout()
        path = os.path.join(output_dir, "actual_vs_predicted.png")
        plt.savefig(path, dpi=150); plt.close()
        log.info(f"Saved → {path}")

        # Residual plot
        residuals = y_test - y_pred
        plt.figure(figsize=(6, 5))
        plt.scatter(y_pred, residuals, alpha=0.5, color="steelblue", edgecolors="white", linewidths=0.3)
        plt.axhline(y=0, color="red", linestyle="--", lw=1.5)
        plt.xlabel("Predicted"); plt.ylabel("Residuals")
        plt.title(f"Residual Plot — {model_name}"); plt.tight_layout()
        path = os.path.join(output_dir, "residual_plot.png")
        plt.savefig(path, dpi=150); plt.close()
        log.info(f"Saved → {path}")

        if HAS_MLFLOW:
            mlflow.log_metric("test_mae", mae)
            mlflow.log_metric("test_rmse", rmse)
            mlflow.log_metric("test_r2", r2)
            mlflow.log_metric("test_mape", mape)

        if HAS_SHAP:
            fn = [f"F{i}" for i in range(X_test.shape[1])]
            explain_with_shap(model, model_name, X_test, fn, output_dir)

        return {"mae": mae, "mse": mse, "rmse": rmse, "r2": r2, "mape": mape}



def build_stacking_ensemble(trained_models, X_train, X_test, y_train, y_test,
                             is_classification, output_dir="eda_output"):
    """
    Build a stacking ensemble from the top 3-5 trained models.
    Uses cross_val_predict for the meta-features to avoid leakage.
    Returns (meta_learner, base_names, stack_score).
    """
    from sklearn.model_selection import cross_val_predict

    banner("Stacking Ensemble")

    # Pick top N models, limit to 5 to keep it fast
    base_names = list(trained_models.keys())[:5]
    log.info(f"Building stacking ensemble from top {len(base_names)} models: {base_names}")

    cv_method = (RepeatedStratifiedKFold(n_splits=3, n_repeats=1, random_state=42)
                 if is_classification else RepeatedKFold(n_splits=3, n_repeats=1, random_state=42))

    # ── Generate meta-features (out-of-fold predictions on train) ─────────────
    meta_train = np.zeros((X_train.shape[0], len(base_names)))
    meta_test  = np.zeros((X_test.shape[0],  len(base_names)))

    for i, name in enumerate(base_names):
        model = trained_models[name]
        log.info(f"  Generating OOF predictions for {name}...")
        if is_classification and hasattr(model, "predict_proba"):
            oof_preds = cross_val_predict(model, X_train, y_train,
                                          cv=cv_method, method="predict_proba")
            if oof_preds.shape[1] == 2:
                meta_train[:, i] = oof_preds[:, 1]
                model.fit(X_train, y_train)
                meta_test[:, i] = model.predict_proba(X_test)[:, 1]
            else:
                meta_train[:, i] = np.argmax(oof_preds, axis=1)
                model.fit(X_train, y_train)
                meta_test[:, i] = model.predict(X_test)
        else:
            meta_train[:, i] = cross_val_predict(model, X_train, y_train, cv=cv_method)
            model.fit(X_train, y_train)
            meta_test[:, i] = model.predict(X_test)

    # ── Train meta-learner on meta-features ────────────────────────────────────
    if is_classification:
        meta_learner = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
        meta_learner.fit(meta_train, y_train)
        stack_preds = meta_learner.predict(meta_test)
        stack_score = f1_score(y_test, stack_preds, average="weighted")
        log.info(f"Stacking Ensemble F1 (weighted): {stack_score:.4f}")
    else:
        meta_learner = Ridge(alpha=1.0)
        meta_learner.fit(meta_train, y_train)
        stack_preds = meta_learner.predict(meta_test)
        stack_score = r2_score(y_test, stack_preds)
        log.info(f"Stacking Ensemble R²: {stack_score:.4f}")

    return meta_learner, base_names, stack_score


def explain_with_shap(model, model_name, X_test, feature_names, output_dir="eda_output"):
    """
    Generate SHAP summary plot for the best model.
    Works best with tree-based models (fast TreeExplainer).
    Falls back to KernelExplainer for others (slow, uses a sample).
    """
    if not HAS_SHAP:
        log.warning("SHAP not installed — skipping explainability. Install: pip install shap")
        return

    section("SHAP Explainability")
    log.info(f"Computing SHAP values for {model_name}...")

    try:
        # TreeExplainer is very fast — works for RF, GBM, XGB, LGB, ET, DT
        tree_based = ("Forest", "Boosting", "Tree", "XGB", "LGBM", "Ada", "Extra")
        if any(kw in model_name for kw in tree_based):
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_test)
        else:
            # KernelExplainer works for any model but is slow — use 100 samples
            X_sample = shap.sample(X_test, min(100, len(X_test)))
            explainer = shap.KernelExplainer(model.predict, X_sample)
            shap_values = explainer.shap_values(X_sample)
            X_test = X_sample  # align for plot

        # For multiclass, shap_values is a list (one array per class)
        # Use class 1 for binary, or the list for multiclass
        sv = shap_values[1] if isinstance(shap_values, list) else shap_values

        plt.figure()
        shap.summary_plot(
            sv, X_test,
            feature_names=feature_names,
            show=False,
            plot_size=(10, 6),
        )
        path = os.path.join(output_dir, "shap_summary.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        log.info(f"SHAP summary plot saved → {path}")

    except Exception as e:
        log.warning(f"SHAP failed for {model_name}: {e}. Skipping explainability.")


# ─── 9. SAVE MODEL ───────────────────────────────────────────────────────────────

def save_model(model, model_name, output_dir="eda_output"):
    """Save the best model + log to MLflow."""
    section("Saving Best Model")
    path = os.path.join(output_dir, "best_model.pkl")
    joblib.dump(model, path)
    log.info(f"Model saved → {os.path.abspath(path)}")
    log.info(f"Load with: joblib.load('{path}')")

    if HAS_MLFLOW:
        mlflow.sklearn.log_model(model, artifact_path="best_model")
        log.info("Model registered in MLflow artifact store")


# ─── 10. MAIN — CLI ENTRY POINT ──────────────────────────────────────────────────

def run_pipeline(file_path, target_col, test_size=0.2, n_trials=50, output_dir="eda_output",
                 task_type="auto", add_interactions=False):
    """
    Core pipeline function — called by both CLI and Streamlit.
    Returns (best_model, best_name, results_df, metrics).
    """
    start = time.time()
    setup_logging(output_dir)

    banner("AutoML Pipeline v2", color=C_GREEN)
    log.info(f"Dataset     : {file_path}")
    log.info(f"Target col  : {target_col}")
    log.info(f"Test size   : {test_size}")
    log.info(f"Optuna trials: {n_trials if HAS_OPTUNA else 'N/A (not installed)'}")
    log.info(f"MLflow      : {'enabled' if HAS_MLFLOW else 'disabled (not installed)'}")
    log.info(f"Output dir  : {output_dir}")

    # Optional: start MLflow experiment
    if HAS_MLFLOW:
        mlflow.set_experiment("AutoML_Pipeline")
        mlflow.start_run(run_name=f"run_{int(time.time())}")
        mlflow.log_param("file_path", file_path)
        mlflow.log_param("target_col", target_col)

    # Step 1 — Load
    df = load_data(file_path, target_col)

    # Step 2 — EDA on RAW data (before any cleaning)
    # This shows the true state of the data: raw distributions, real missing values, real outliers
    run_eda(df, target_col, output_dir=output_dir)

    # Step 3 — Clean (after EDA so EDA reflects ground truth)
    df = clean_data(df, target_col, output_dir=output_dir)

    # Step 3 — Prepare data (separate X/y, encode target)
    X, y, numeric_cols, low_card_cols, high_card_cols, skewed_cols, target_le, text_cols = prepare_data(df, target_col)
    cat_cols = low_card_cols + high_card_cols

    # Detect problem type
    section("Problem Type Detection")
    if task_type != "auto":
        is_classification = (task_type == "classification")
        log.info(f"Task type FORCED by user: {task_type.upper()}")
    else:
        n_unique = y.nunique()
        unique_ratio = n_unique / len(y)

        if n_unique == 2:
            is_classification = True
            log.info("2 unique target values → binary CLASSIFICATION")
        elif n_unique <= 20 and unique_ratio < 0.05:
            is_classification = True
            log.info(f"{n_unique} unique values, ratio={unique_ratio:.3f} → CLASSIFICATION")
        elif pd.api.types.is_float_dtype(y) and n_unique > 20:
            is_classification = False
            log.info(f"Float target with {n_unique} unique values → REGRESSION")
        elif n_unique > 50 or unique_ratio > 0.10:
            is_classification = False
            log.info(f"{n_unique} unique values, ratio={unique_ratio:.3f} → REGRESSION")
        else:
            is_classification = True
            log.warning(f"Ambiguous target ({n_unique} unique, ratio={unique_ratio:.3f}). "
                        f"Defaulting to CLASSIFICATION. Use --task-type regression to override.")

    task_type_str = "CLASSIFICATION" if is_classification else "REGRESSION"
    log.info(f"Detected: {B}{task_type_str}{END}")

    if HAS_MLFLOW:
        mlflow.log_param("task_type", task_type_str)

    # Step 4 — Build preprocessing pipeline (NO data leakage!)
    section("Building sklearn Pipeline")
    preprocessor = build_preprocessor(
        numeric_cols, low_card_cols, high_card_cols, skewed_cols,
        add_interactions=add_interactions, text_cols=text_cols
    )
    if add_interactions:
        log.info("Polynomial interaction features ENABLED (--interactions flag)")
    log.info("Preprocessing will be fit ONLY on training data (no leakage)")
        
        
    # Remove classes with fewer than 2 samples (can't stratify on them)
    from collections import Counter
    if is_classification:
        class_counts = Counter(y)
        rare_classes = [cls for cls, count in class_counts.items() if count < 2]
        if rare_classes:
            log.warning(f"Dropping {len(rare_classes)} rare class(es) with only 1 sample: {rare_classes}")
            mask = ~y.isin(rare_classes)
            X = X[mask]
            y = y[mask]
            log.info(f"Dataset size after rare class removal: {X.shape[0]} rows")
        
    # Step 5 — Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42,
        stratify=y if is_classification else None,
    )

    # Step 6 — Fit preprocessor on train, transform both
    section("Fitting Preprocessor on Training Data")
    X_train_t = preprocessor.fit_transform(X_train, y_train)
    X_test_t = preprocessor.transform(X_test)
    log.info(f"Transformed shapes: train={X_train_t.shape}, test={X_test_t.shape}")

    # Convert sparse matrices if needed
    if hasattr(X_train_t, "toarray"):
        X_train_t = X_train_t.toarray()
    if hasattr(X_test_t, "toarray"):
        X_test_t = X_test_t.toarray()

    if hasattr(preprocessor, "get_feature_names_out"):
        try:
            fn = preprocessor.get_feature_names_out()
        except:
            fn = None
    else:
        fn = None

    # Step 7 — Feature selection (on transformed data)
    selector = select_features_from_transformed(X_train_t, y_train, fn, is_classification, output_dir)
    if selector is not None:
        X_train_t = selector.transform(X_train_t)
        X_test_t = selector.transform(X_test_t)
        log.info(f"After selection: train={X_train_t.shape}, test={X_test_t.shape}")

    X_train_t, y_train, imbalanced = handle_class_imbalance(X_train_t, y_train, is_classification)

    # Step 8 — Train models (with Optuna + Repeated K-Fold)
    best_model, best_name, results_df, best_cv_score = train_models(
        X_train_t, X_test_t, y_train, y_test,
        is_classification, n_trials=n_trials, output_dir=output_dir, imbalanced=imbalanced
    )

    # Step 8.5 — Stacking Ensemble (compare against best single model)
    try:
        # Re-train all models on full training set for stacking base models
        all_defaults = get_default_models(is_classification)
        trained_for_stack = {}
        for nm, mdl in all_defaults.items():
            mdl.fit(X_train_t, y_train)
            trained_for_stack[nm] = mdl

        _, _, stack_score = build_stacking_ensemble(
            trained_for_stack, X_train_t, X_test_t, y_train, y_test,
            is_classification, output_dir=output_dir
        )
        cv_metric = "F1" if is_classification else "R²"
        log.info(f"Stacking score ({cv_metric}): {stack_score:.4f}  vs  Best single model CV: {best_cv_score:.4f}")
        if stack_score > best_cv_score:
            log.info("Stacking ensemble outperforms best single model — consider using it for deployment")
        else:
            log.info("Best single model holds — stacking did not improve performance")
    except Exception as e:
        log.warning(f"Stacking ensemble failed: {e}. Proceeding with best single model.")

    # Step 9 — Evaluate best model
    metrics = evaluate(best_model, best_name, X_test_t, y_test,
                       is_classification, target_le, best_cv_score=best_cv_score, output_dir=output_dir)

    # Step 10 — Save
    # Save the full pipeline (preprocessor + selector + model) for deployment
    full_pipeline = {
        "preprocessor": preprocessor,
        "selector": selector,
        "model": best_model,
        "model_name": best_name,
        "target_le": target_le,
        "is_classification": is_classification,
        "numeric_cols": numeric_cols,
        "cat_cols": cat_cols,
        "metrics": metrics,
    }
    section("Saving Full Pipeline")
    path = os.path.join(output_dir, "full_pipeline.pkl")
    joblib.dump(full_pipeline, path)
    log.info(f"Full pipeline → {os.path.abspath(path)}")

    save_model(best_model, best_name, output_dir=output_dir)

    # Finalize MLflow
    if HAS_MLFLOW:
        mlflow.end_run()
        log.info("MLflow run ended — check UI with: mlflow ui")

    # ── Done ──────────────────────────────────────────────────────────────────
    elapsed = time.time() - start
    banner("Pipeline Complete", color=C_GREEN)
    log.info(f"Total time: {elapsed:.1f}s")
    log.info(f"Best model: {best_name}")
    log.info(f"Outputs in: {os.path.abspath(output_dir)}/")
    print(f"\n  {B}{C_CYAN}AutoML Pipeline v2 — Complete! 🚀{END}\n")

    return best_model, best_name, results_df, metrics


def predict_new(pipeline_path, data_path, output_path="predictions.csv"):
    """Load an existing pipeline artifact and run inference on new data."""
    setup_logging(os.path.dirname(output_path) or ".")
    banner("Inference Mode")
    if not os.path.isfile(pipeline_path):
        log.error(f"Pipeline not found: {pipeline_path}")
        return

    ext = os.path.splitext(data_path)[1].lower()
    if ext == ".csv": new_df = pd.read_csv(data_path)
    elif ext in [".xls", ".xlsx"]: new_df = pd.read_excel(data_path)
    elif ext == ".parquet": new_df = pd.read_parquet(data_path)
    elif ext == ".json": new_df = pd.read_json(data_path)
    else: log.error("Unsupported file"); return

    log.info("Loading pipeline...")
    pipeline = joblib.load(pipeline_path)
    preprocessor = pipeline["preprocessor"]
    selector = pipeline["selector"]
    model = pipeline["model"]
    target_le = pipeline.get("target_le")
    is_classification = pipeline["is_classification"]

    # ── Column Validation ─────────────────────────────────────────────────────
    expected_numeric = pipeline.get("numeric_cols", [])
    expected_cats = pipeline.get("cat_cols", [])
    expected_cols = expected_numeric + expected_cats

    missing_cols = [c for c in expected_cols if c not in new_df.columns]
    extra_cols = [c for c in new_df.columns if c not in expected_cols]

    if missing_cols:
        log.error(f"Input data is missing {len(missing_cols)} required column(s): {missing_cols}")
        log.error("These columns were present during training. Cannot proceed.")
        return None

    if extra_cols:
        log.warning(f"Input data has {len(extra_cols)} extra column(s) not seen during training: {extra_cols}")
        log.warning("These will be ignored by the preprocessor (remainder='drop').")

    log.info(f"Column validation passed — {len(expected_cols)} expected columns found.")
    # ──────────────────────────────────────────────────────────────────────────

    log.info("Applying preprocessing...")
    X_t = preprocessor.transform(new_df)
    if hasattr(X_t, "toarray"): X_t = X_t.toarray()

    if selector:
        log.info("Applying feature selection...")
        X_t = selector.transform(X_t)

    log.info("Predicting...")
    preds = model.predict(X_t)
    
    out_df = new_df.copy()
    if is_classification and target_le is not None:
        out_df["Prediction"] = target_le.inverse_transform(preds)
    else:
        out_df["Prediction"] = preds
    
    if is_classification and hasattr(model, "predict_proba"):
        probas = model.predict_proba(X_t)
        if probas.shape[1] == 2:
            out_df["Probability_1"] = probas[:, 1]
        else:
            for i in range(probas.shape[1]):
                out_df[f"Probability_Class_{i}"] = probas[:, i]

    out_df.to_csv(output_path, index=False)
    log.info(f"Predictions saved to {output_path}")
    return out_df


def main():
    parser = argparse.ArgumentParser(
        description="AutoML Pipeline v2 — Production-Grade Model Builder",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Example:\n  python first.py data.csv price --test-size 0.25\n  python first.py --predict new_data.csv --model eda_output/full_pipeline.pkl",
    )
    parser.add_argument("file_path", nargs="?", help="Path to the dataset (CSV, Excel, Parquet, JSON)")
    parser.add_argument("target", nargs="?", help="Name of the target column (ignored in predict mode)")
    parser.add_argument("--predict", action="store_true", help="Run inference mode")
    parser.add_argument("--model", default="eda_output/full_pipeline.pkl", help="Path to full_pipeline.pkl")
    parser.add_argument("--test-size", type=float, default=0.2, dest="test_size",
                        help="Test split ratio (default: 0.2)")
    parser.add_argument("--trials", type=int, default=50,
                        help="Number of Optuna trials per model (default: 50, 0 to disable)")
    parser.add_argument("--output", default="eda_output",
                        help="Output directory for plots, models, logs (default: eda_output)")
    parser.add_argument(
        "--task-type",
        choices=["auto", "classification", "regression"],
        default="auto",
        dest="task_type",
        help="Force task type instead of auto-detecting (default: auto)",
    )
    parser.add_argument(
        "--interactions",
        action="store_true",
        default=False,
        help="Add pairwise interaction features for numeric columns (default: off)",
    )
    args = parser.parse_args()

    if args.predict:
        if not args.file_path:
            log.error("Please provide the file_path for prediction.")
            sys.exit(1)
        predict_new(args.model, args.file_path, os.path.join(args.output, "predictions.csv"))
    else:
        if not args.file_path or not args.target:
            log.error("Please provide both file_path and target column for training.")
            sys.exit(1)
        run_pipeline(
            file_path=args.file_path,
            target_col=args.target,
            test_size=args.test_size,
            n_trials=args.trials,
            output_dir=args.output,
            task_type=args.task_type,
            add_interactions=args.interactions,
        )

if __name__ == "__main__":
    main()
