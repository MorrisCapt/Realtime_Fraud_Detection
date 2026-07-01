"""
train.py — Full training pipeline for Umba fraud detection.

Usage:
    python train.py

Outputs:
    model/lgbm_model.pkl    — trained LightGBM model + metadata
    model/val_metrics.json  — validation metrics (PR-AUC, ROC-AUC, operational recall)
"""

import json
import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    precision_recall_curve,
)

from preprocess import preprocess, get_feature_columns

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATA_DIR = Path("data")
MODEL_DIR = Path("model")
MODEL_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_data():
    print("Loading data...")
    train = pd.read_csv(DATA_DIR / "train.csv")
    identity = pd.read_csv(DATA_DIR / "identity.csv")
    total = len(train)
    fraud_n = train["isFraud"].sum()
    print(f"  Train rows : {total:,}")
    print(f"  Fraud rows : {fraud_n:,}  ({fraud_n/total*100:.2f}%)")
    print(f"  Identity rows : {len(identity):,}  |  unique TxnIDs: {identity['TransactionID'].nunique():,}")
    return train, identity


def time_split(df: pd.DataFrame, val_fraction: float = 0.20):
    """
    Validation set = the LAST val_fraction of the time range.

    We must NOT do a random split because test.csv covers a later period
    than train.csv.  A random split would let future information leak into
    training and inflate validation scores.
    """
    dt_min = df["TransactionDT"].min()
    dt_max = df["TransactionDT"].max()
    split_dt = dt_min + (1 - val_fraction) * (dt_max - dt_min)

    train_mask = df["TransactionDT"] < split_dt
    val_mask   = df["TransactionDT"] >= split_dt

    print(f"\nTime-based split  (split DT = {split_dt:.0f})")
    print(f"  Training rows   : {train_mask.sum():,}  "
          f"(fraud rate {df.loc[train_mask,'isFraud'].mean()*100:.2f}%)")
    print(f"  Validation rows : {val_mask.sum():,}  "
          f"(fraud rate {df.loc[val_mask,'isFraud'].mean()*100:.2f}%)")

    return train_mask, val_mask


def build_lgbm_datasets(X_tr, y_tr, X_val, y_val, cat_cols):
    dtrain = lgb.Dataset(X_tr, label=y_tr,
                         categorical_feature=cat_cols,
                         free_raw_data=False)
    dval   = lgb.Dataset(X_val, label=y_val,
                         categorical_feature=cat_cols,
                         reference=dtrain,
                         free_raw_data=False)
    return dtrain, dval


def train_model(X_tr, y_tr, X_val, y_val, cat_cols):
    """
    Train LightGBM binary classifier.

    Imbalance strategy: scale_pos_weight = negatives / positives.
    This re-weights the loss so rare fraud events have equal total weight
    to the majority class.  We also use PR-AUC as the early-stopping metric
    because it is the honest metric under heavy class imbalance — ROC-AUC
    can look great even when precision at high recall is terrible.
    """
    neg = int((y_tr == 0).sum())
    pos = int((y_tr == 1).sum())
    spw = neg / pos
    print(f"\n  Imbalance ratio  : {spw:.1f}x  (neg={neg:,}, pos={pos:,})")
    print(f"  scale_pos_weight : {spw:.1f}")

    params = {
        "objective":         "binary",
        "metric":            "average_precision",   # PR-AUC tracked on val
        "scale_pos_weight":  spw,
        "learning_rate":     0.05,
        "num_leaves":        63,
        "min_child_samples": 50,       # prevents overfitting on tiny fraud clusters
        "feature_fraction":  0.8,
        "bagging_fraction":  0.8,
        "bagging_freq":      5,
        "reg_alpha":         0.1,
        "reg_lambda":        0.1,
        "verbose":           -1,
        "n_jobs":            -1,
        "seed":              42,
    }

    dtrain, dval = build_lgbm_datasets(X_tr, y_tr, X_val, y_val, cat_cols)

    callbacks = [
        lgb.early_stopping(stopping_rounds=50, verbose=True),
        lgb.log_evaluation(period=100),
    ]

    model = lgb.train(
        params,
        dtrain,
        num_boost_round=1000,
        valid_sets=[dtrain, dval],
        valid_names=["train", "val"],
        callbacks=callbacks,
    )

    return model


def evaluate(model, X_val, y_val, feature_names):
    """
    Report PR-AUC (primary), ROC-AUC, and operational recall.

    Operational recall answers the question Umba actually cares about:
    'If our ops team can only manually review the top X% of riskiest
    transactions flagged by the model, what fraction of all fraud do they catch?'
    We report X = 5% and X = 10%.
    """
    probs = model.predict(X_val)
    total_fraud = int(y_val.sum())

    pr_auc  = average_precision_score(y_val, probs)
    roc_auc = roc_auc_score(y_val, probs)

    def recall_at_top_k(k_frac):
        k = max(1, int(k_frac * len(y_val)))
        top_idx = np.argsort(probs)[::-1][:k]
        caught = int(y_val.iloc[top_idx].sum())
        return caught / total_fraud if total_fraud > 0 else 0, caught

    rec_5,  caught_5  = recall_at_top_k(0.05)
    rec_10, caught_10 = recall_at_top_k(0.10)

    # Best F1 threshold
    precision_arr, recall_arr, thresholds_arr = precision_recall_curve(y_val, probs)
    f1_arr = (2 * precision_arr * recall_arr
              / (precision_arr + recall_arr + 1e-9))
    best_idx = int(np.argmax(f1_arr))
    best_threshold = float(thresholds_arr[best_idx]) if best_idx < len(thresholds_arr) else 0.5
    best_f1        = float(f1_arr[best_idx])

    print("\n=== Validation Metrics ===")
    print(f"  PR-AUC  (primary)           : {pr_auc:.4f}")
    print(f"  ROC-AUC                     : {roc_auc:.4f}")
    print(f"  Recall @ top  5% reviewed   : {rec_5:.4f}  ({caught_5}/{total_fraud} fraud caught)")
    print(f"  Recall @ top 10% reviewed   : {rec_10:.4f}  ({caught_10}/{total_fraud} fraud caught)")
    print(f"  Calibration — mean pred prob: {probs.mean():.4f}  (actual rate: {y_val.mean():.4f})")
    print(f"  Best F1 threshold           : {best_threshold:.4f}  (F1 = {best_f1:.4f})")

    # Top-20 features by split gain
    importance = (
        pd.Series(model.feature_importance(importance_type="gain"),
                  index=feature_names)
        .sort_values(ascending=False)
    )
    print("\n=== Top 20 Features (by gain) ===")
    print(importance.head(20).to_string())

    metrics = {
        "pr_auc":               round(float(pr_auc), 4),
        "roc_auc":              round(float(roc_auc), 4),
        "recall_at_top_5pct":   round(float(rec_5), 4),
        "recall_at_top_10pct":  round(float(rec_10), 4),
        "mean_predicted_prob":  round(float(probs.mean()), 4),
        "actual_fraud_rate":    round(float(y_val.mean()), 4),
        "best_threshold":       round(best_threshold, 4),
        "best_f1":              round(best_f1, 4),
        "val_size":             int(len(y_val)),
        "val_fraud_count":      total_fraud,
    }
    return metrics


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("Umba Fraud Detection  —  Training Pipeline")
    print("=" * 60)

    # 1. Load raw data
    train_raw, identity = load_data()

    # 2. Preprocess
    #    - Drops flagged_for_review (leakage)
    #    - Encodes M columns
    #    - Engineers features (log-amt, time signals, email fixes)
    #    - Joins aggregated identity
    print("\nPreprocessing...")
    df = preprocess(train_raw, identity)

    # 3. Time-based train / val split
    train_mask, val_mask = time_split(df)

    # 4. Build feature matrix
    num_cols, cat_cols_all = get_feature_columns()
    all_feature_cols = num_cols + cat_cols_all

    # Keep only columns that survived preprocessing
    all_feature_cols = [c for c in all_feature_cols if c in df.columns]
    cat_cols         = [c for c in cat_cols_all if c in df.columns]

    X_tr  = df.loc[train_mask, all_feature_cols].copy()
    y_tr  = df.loc[train_mask, "isFraud"].copy()
    X_val = df.loc[val_mask,   all_feature_cols].copy()
    y_val = df.loc[val_mask,   "isFraud"].copy()

    print(f"\nFeature matrix  : {len(all_feature_cols)} features "
          f"({len(cat_cols)} categorical, {len(all_feature_cols)-len(cat_cols)} numeric)")

    # 5. Train
    print("\nTraining LightGBM...")
    model = train_model(X_tr, y_tr, X_val, y_val, cat_cols)

    # 6. Evaluate
    print("\nEvaluating on validation set...")
    metrics = evaluate(model, X_val, y_val, all_feature_cols)

    # 7. Save artefact  (model + feature list + metrics bundled together so
    #    predict.py and the API always use exactly the same feature set)
    artifact = {
        "model":        model,
        "feature_cols": all_feature_cols,
        "cat_cols":     cat_cols,
        "metrics":      metrics,
    }

    model_path   = MODEL_DIR / "lgbm_model.pkl"
    metrics_path = MODEL_DIR / "val_metrics.json"

    with open(model_path, "wb") as f:
        pickle.dump(artifact, f)

    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\nModel   saved → {model_path}")
    print(f"Metrics saved → {metrics_path}")
    print("\nAll done!  Run `python predict.py` to score test.csv")


if __name__ == "__main__":
    main()
