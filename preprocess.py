"""
preprocess.py — Shared preprocessing logic.

Imported by train.py, predict.py, and api/main.py so that the exact same
transformations are applied at training time and at inference time.
"""

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

EMAIL_FIXES = {
    "gmail.con": "gmail.com",
    "gmial.com": "gmail.com",
    "ymail.con": "yahoo.com",
    "hotmail.con": "hotmail.com",
    "yahoo.con": "yahoo.com",
}

M_COLS = ["M1", "M2", "M3", "M4", "M5", "M6"]
C_COLS = ["C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8"]
D_COLS = ["D1", "D2", "D3", "D4", "D5"]
V_COLS = [f"V{i}" for i in range(1, 21)]
ID_COLS = [f"id_{i:02d}" for i in range(1, 12)]

CAT_COLS = [
    "country", "currency", "channel", "card_type", "card_bank",
    "P_emaildomain", "R_emaildomain", "DeviceType",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _col(df, col, default=np.nan):
    """Safely get a column; return a Series of default if absent."""
    if col in df.columns:
        return df[col]
    return pd.Series([default] * len(df), index=df.index)


def _fix_email(domain) -> str:
    if pd.isna(domain):
        return "unknown"
    domain = str(domain).strip().lower()
    return EMAIL_FIXES.get(domain, domain)


def aggregate_identity(identity_df: pd.DataFrame) -> pd.DataFrame:
    """
    Collapse identity.csv to one row per TransactionID.

    The file is NOT 1-to-1: some transactions have multiple device sessions.
    We average numeric id_* columns, take mode of DeviceType, first DeviceInfo,
    and add session_count + has_identity=1.
    """
    id_num_cols = [c for c in ID_COLS if c in identity_df.columns]

    agg_dict = {col: "mean" for col in id_num_cols}
    agg_dict["DeviceType"] = lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else np.nan
    agg_dict["DeviceInfo"] = "first"

    agg = identity_df.groupby("TransactionID").agg(agg_dict).reset_index()
    session_counts = (
        identity_df.groupby("TransactionID")
        .size()
        .reset_index(name="session_count")
    )
    agg = agg.merge(session_counts, on="TransactionID", how="left")
    agg["has_identity"] = 1
    return agg


def _encode_m_cols(df: pd.DataFrame) -> pd.DataFrame:
    """T -> 1, F -> 0, NaN stays NaN."""
    df = df.copy()
    for col in M_COLS:
        if col in df.columns:
            df[col] = df[col].map({"T": 1, "F": 0})
    return df


def _engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Derive new features from raw columns.
    Uses _col() for optional fields so single-row API requests don't crash
    when only mandatory fields are supplied.
    """
    df = df.copy()

    df["amt_log"] = np.log1p(df["TransactionAmt"])
    df["hour_of_day"] = (df["TransactionDT"] // 3600) % 24
    df["day_of_week"] = (df["TransactionDT"] // 86400) % 7

    # Email domain cleanup (handles missing columns safely)
    p_email = _col(df, "P_emaildomain", None).apply(_fix_email)
    r_email = _col(df, "R_emaildomain", None).apply(_fix_email)
    df["P_emaildomain"] = p_email
    df["R_emaildomain"] = r_email
    df["email_domain_match"] = (p_email == r_email).astype(int)

    # Account-age and velocity signals (safe on missing columns)
    df["recipient_is_new"] = (
        _col(df, "recipient_account_age_days", 9999).fillna(9999) < 30
    ).astype(int)

    df["sender_is_new"] = (
        _col(df, "sender_prev_txn_count", 1).fillna(1) == 0
    ).astype(int)

    return df


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_feature_columns():
    """Return (numeric_cols, categorical_cols)."""
    num_cols = (
        [
            "TransactionAmt", "amt_log",
            "card1", "card2", "card3", "card5",
            "addr1", "addr2", "dist1", "dist2",
            "recipient_account_age_days", "sender_prev_txn_count",
            "hour_of_day", "day_of_week",
            "email_domain_match", "recipient_is_new", "sender_is_new",
            "has_identity", "session_count",
        ]
        + C_COLS + D_COLS + M_COLS + V_COLS + ID_COLS
    )
    return num_cols, CAT_COLS


def preprocess(df: pd.DataFrame, identity_df: pd.DataFrame = None) -> pd.DataFrame:
    """
    Full preprocessing pipeline — identical at train and inference time.

    1. Drop flagged_for_review (post-review leakage field — never available at score time).
    2. Encode M columns (T/F -> 1/0).
    3. Engineer features (log-amt, time signals, email fixes, account-age flags).
    4. Join aggregated identity (if provided).
    5. Cast categoricals to pandas 'category' dtype for LightGBM.
    """
    df = df.copy()

    # 1. Drop leakage
    df = df.drop(columns=["flagged_for_review"], errors="ignore")

    # 2. Encode M columns
    df = _encode_m_cols(df)

    # 3. Feature engineering
    df = _engineer_features(df)

    # 4. Merge identity
    if identity_df is not None:
        agg = aggregate_identity(identity_df)
        df = df.merge(agg, on="TransactionID", how="left")
        df["has_identity"] = df["has_identity"].fillna(0).astype(int)
        df["session_count"] = df["session_count"].fillna(0).astype(int)
    else:
        df["has_identity"] = 0
        df["session_count"] = 0

    # 5. Categorical columns — fill NaN, cast to category dtype
    for col in CAT_COLS:
        if col in df.columns:
            df[col] = df[col].fillna("unknown").astype(str).astype("category")
        else:
            df[col] = pd.Categorical(["unknown"] * len(df))

    return df
