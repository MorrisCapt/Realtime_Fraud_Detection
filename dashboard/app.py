"""
dashboard/app.py — Streamlit operations dashboard for Umba Fraud Detection.

Usage:
    streamlit run dashboard/app.py

What it shows:
    - Key metrics: total transactions, flagged count, mean fraud score, val PR-AUC
    - Score distribution with adjustable alarm threshold
    - Flagged rate by channel and country
    - Top flagged transactions table
    - Validation metrics from training
"""

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

# Allow imports from project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Umba Fraud Detection",
    page_icon="🛡️",
    layout="wide",
)

ROOT        = Path(__file__).resolve().parent.parent
DATA_DIR    = ROOT / "data"
MODEL_DIR   = ROOT / "model"
PRED_PATH   = ROOT / "predictions.csv"


# ---------------------------------------------------------------------------
# Data loaders (cached so Streamlit doesn't reload on every widget interaction)
# ---------------------------------------------------------------------------

@st.cache_data
def load_data():
    test        = pd.read_csv(DATA_DIR / "test.csv")
    predictions = pd.read_csv(PRED_PATH)
    df = test.merge(predictions, on="TransactionID", how="left")
    return df


@st.cache_data
def load_metrics():
    path = MODEL_DIR / "val_metrics.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {}


# ---------------------------------------------------------------------------
# Guard: files must exist before we render anything
# ---------------------------------------------------------------------------

missing_files = []
for p in [PRED_PATH, DATA_DIR / "test.csv", MODEL_DIR / "val_metrics.json"]:
    if not p.exists():
        missing_files.append(str(p))

if missing_files:
    st.error(
        "The following required files are missing:\n\n"
        + "\n".join(f"- `{f}`" for f in missing_files)
        + "\n\nRun `python train.py` then `python predict.py` first."
    )
    st.stop()

# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------

df      = load_data()
metrics = load_metrics()

# ---------------------------------------------------------------------------
# Sidebar — threshold control
# ---------------------------------------------------------------------------

st.sidebar.image("https://umba.com/favicon.ico", width=40) if False else None
st.sidebar.title("🛡️ Umba Fraud Detection")
st.sidebar.markdown("---")

default_threshold = float(metrics.get("best_threshold", 0.5))
threshold = st.sidebar.slider(
    "Alarm threshold",
    min_value=0.0, max_value=1.0,
    value=default_threshold, step=0.01,
    help="Transactions with fraud_prob ≥ threshold are flagged for ops review.",
)

st.sidebar.markdown("---")
st.sidebar.markdown(
    f"**Best-F1 threshold** (from training): `{default_threshold}`\n\n"
    f"Adjust above to trade off precision vs recall for your ops capacity."
)

df["alarm"] = df["isFraud_prob"] >= threshold

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------

st.title("🛡️ Umba Fraud Detection — Operations Dashboard")
st.caption(
    "Model: LightGBM  |  Primary metric: PR-AUC  |  "
    "Markets: Kenya 🇰🇪 & Nigeria 🇳🇬"
)

# ---------------------------------------------------------------------------
# KPI row
# ---------------------------------------------------------------------------

total   = len(df)
flagged = int(df["alarm"].sum())
mean_score = df["isFraud_prob"].mean()

k1, k2, k3, k4 = st.columns(4)
k1.metric("Total Transactions",  f"{total:,}")
k2.metric("Flagged for Review",  f"{flagged:,}",
          delta=f"{flagged/total*100:.1f}% of all")
k3.metric("Mean Fraud Score",    f"{mean_score*100:.3f}%")
k4.metric("Val PR-AUC",          str(metrics.get("pr_auc", "N/A")))

st.markdown("---")

# ---------------------------------------------------------------------------
# Row 2  — Score distribution  |  Flagged rate by channel
# ---------------------------------------------------------------------------

col1, col2 = st.columns(2)

with col1:
    st.subheader("Score Distribution")
    fig, ax = plt.subplots(figsize=(6, 3.2))
    ax.hist(df["isFraud_prob"], bins=60, color="#2563eb", edgecolor="white",
            alpha=0.85, linewidth=0.4)
    ax.axvline(threshold, color="#ef4444", linestyle="--", linewidth=1.5,
               label=f"Threshold = {threshold:.2f}")
    ax.set_xlabel("Fraud Probability", fontsize=10)
    ax.set_ylabel("Transaction Count", fontsize=10)
    ax.legend(fontsize=9)
    ax.set_title("Distribution of Predicted Fraud Scores", fontsize=11)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close()

with col2:
    st.subheader("Flag Rate by Channel")
    ch = (
        df.groupby("channel")
        .agg(total=("TransactionID", "count"), flagged=("alarm", "sum"))
        .assign(flag_rate=lambda d: d["flagged"] / d["total"])
        .sort_values("flag_rate", ascending=False)
        .reset_index()
    )
    fig, ax = plt.subplots(figsize=(6, 3.2))
    colors = ["#ef4444" if r > 0.06 else "#f97316" if r > 0.02 else "#22c55e"
              for r in ch["flag_rate"]]
    ax.barh(ch["channel"], ch["flag_rate"] * 100, color=colors, edgecolor="white")
    ax.set_xlabel("Flagged (%)", fontsize=10)
    ax.set_title("Proportion Flagged per Channel", fontsize=11)
    for i, (val, total_n) in enumerate(zip(ch["flag_rate"], ch["total"])):
        ax.text(val * 100 + 0.1, i, f"{val*100:.1f}%  (n={total_n:,})",
                va="center", fontsize=8)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close()

# ---------------------------------------------------------------------------
# Row 3 — Country breakdown  |  Card type breakdown
# ---------------------------------------------------------------------------

col1, col2 = st.columns(2)

with col1:
    st.subheader("Breakdown by Country")
    country_stats = (
        df.groupby("country")
        .agg(
            Transactions=("TransactionID", "count"),
            Flagged=("alarm", "sum"),
            Mean_Score=("isFraud_prob", "mean"),
        )
        .assign(Flag_Rate=lambda d: d["Flagged"] / d["Transactions"])
        .reset_index()
        .rename(columns={"country": "Country"})
    )
    st.dataframe(
        country_stats.style.format({
            "Flag_Rate":  "{:.2%}",
            "Mean_Score": "{:.4f}",
        }),
        use_container_width=True,
    )

with col2:
    st.subheader("Breakdown by Card Type")
    card_stats = (
        df.groupby("card_type")
        .agg(
            Transactions=("TransactionID", "count"),
            Flagged=("alarm", "sum"),
            Mean_Score=("isFraud_prob", "mean"),
        )
        .assign(Flag_Rate=lambda d: d["Flagged"] / d["Transactions"])
        .reset_index()
        .rename(columns={"card_type": "Card Type"})
    )
    st.dataframe(
        card_stats.style.format({
            "Flag_Rate":  "{:.2%}",
            "Mean_Score": "{:.4f}",
        }),
        use_container_width=True,
    )

st.markdown("---")

# ---------------------------------------------------------------------------
# Top flagged transactions
# ---------------------------------------------------------------------------

st.subheader(f"🚨 Top Flagged Transactions  (threshold = {threshold:.2f})")

display_cols = [
    "TransactionID", "TransactionAmt", "currency", "country",
    "channel", "card_type", "card_bank", "isFraud_prob",
]
top_flagged = (
    df[df["alarm"]][display_cols]
    .sort_values("isFraud_prob", ascending=False)
    .head(100)
    .reset_index(drop=True)
)

if top_flagged.empty:
    st.info("No transactions flagged at the current threshold.")
else:
    st.dataframe(
        top_flagged.style.format({
            "isFraud_prob":  "{:.4f}",
            "TransactionAmt": "{:,.2f}",
        }).background_gradient(subset=["isFraud_prob"], cmap="Reds"),
        use_container_width=True,
    )

st.markdown("---")

# ---------------------------------------------------------------------------
# Validation metrics (from training)
# ---------------------------------------------------------------------------

if metrics:
    st.subheader("📊 Model Validation Metrics  (time-based holdout)")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("PR-AUC",             metrics.get("pr_auc", "N/A"))
    m2.metric("ROC-AUC",            metrics.get("roc_auc", "N/A"))
    m3.metric("Recall @ Top 5%",   f"{metrics.get('recall_at_top_5pct', 0)*100:.1f}%")
    m4.metric("Recall @ Top 10%",  f"{metrics.get('recall_at_top_10pct', 0)*100:.1f}%")

    st.caption(
        f"Validation set: {metrics.get('val_size', 'N/A'):,} transactions  |  "
        f"Fraud count: {metrics.get('val_fraud_count', 'N/A')}  |  "
        f"Best F1 threshold: {metrics.get('best_threshold')}  |  "
        f"Best F1: {metrics.get('best_f1')}"
    )

st.markdown("---")
st.caption("Umba Microfinance Bank — Fraud Detection v1  |  Morris Maina")
