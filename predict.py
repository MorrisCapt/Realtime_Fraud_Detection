"""
predict.py — Score test.csv and produce predictions.csv.

Usage:
    python predict.py

Output:
    predictions.csv  —  TransactionID, isFraud_prob
                        Matches sample_submission.csv format exactly.
"""

import pickle
import warnings
from pathlib import Path

import pandas as pd

from preprocess import preprocess

warnings.filterwarnings("ignore")

DATA_DIR  = Path("data")
MODEL_DIR = Path("model")


def main():
    # 1. Load model artefact
    model_path = MODEL_DIR / "lgbm_model.pkl"
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found at {model_path}. "
            "Run `python train.py` first."
        )

    print("Loading model...")
    with open(model_path, "rb") as f:
        artifact = pickle.load(f)

    model        = artifact["model"]
    feature_cols = artifact["feature_cols"]

    print(f"  Val PR-AUC: {artifact['metrics'].get('pr_auc', 'N/A')}")

    # 2. Load test data
    print("\nLoading test data...")
    test     = pd.read_csv(DATA_DIR / "test.csv")
    identity = pd.read_csv(DATA_DIR / "identity.csv")
    print(f"  Test rows     : {len(test):,}")
    print(f"  Identity rows : {len(identity):,}")

    # 3. Preprocess (same pipeline as training — drops flagged_for_review,
    #    encodes M cols, engineers features, joins identity)
    print("\nPreprocessing...")
    df = preprocess(test, identity)

    # 4. Align feature columns (only keep what the model was trained on)
    # Ensure all training features are present (pad absent columns with NaN)
    feature_cols_present = [c for c in feature_cols if c in df.columns]
    missing = set(feature_cols) - set(feature_cols_present)
    if missing:
        print(f"  Note: {len(missing)} training features absent in test — padded with NaN")
        for col in missing:
            df[col] = np.nan

    X_test = df[feature_cols].copy()

    # 5. Score
    print("\nScoring...")
    probs = model.predict(X_test)

    # 6. Build submission dataframe
    submission = pd.DataFrame({
        "TransactionID": test["TransactionID"],
        "isFraud_prob":  probs,
    })

    # 7. Validate against sample_submission format
    sample = pd.read_csv(DATA_DIR / "sample_submission.csv")
    sample_ids = set(sample["TransactionID"])
    pred_ids   = set(submission["TransactionID"])

    if sample_ids != pred_ids:
        extra   = pred_ids - sample_ids
        missing = sample_ids - pred_ids
        raise ValueError(
            f"TransactionID mismatch!  "
            f"Extra in predictions: {len(extra)}, "
            f"Missing from predictions: {len(missing)}"
        )

    # Sort by TransactionID to match sample_submission order
    submission = submission.sort_values("TransactionID").reset_index(drop=True)

    out_path = Path("predictions.csv")
    submission.to_csv(out_path, index=False)

    # 8. Summary
    flagged_50  = (probs >= 0.50).sum()
    flagged_thr = (probs >= artifact["metrics"].get("best_threshold", 0.5)).sum()

    print(f"\nSaved → {out_path}  ({len(submission):,} rows)")
    print(f"  Mean fraud probability         : {probs.mean()*100:.3f}%")
    print(f"  Transactions flagged (p ≥ 0.50): {flagged_50:,}")
    print(f"  Transactions flagged (best-F1 threshold "
          f"{artifact['metrics'].get('best_threshold', 0.5):.3f}): {flagged_thr:,}")


if __name__ == "__main__":
    main()
