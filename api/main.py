"""
api/main.py — FastAPI fraud-scoring service.

Usage:
    uvicorn api.main:app --reload --port 8000

Endpoints:
    GET  /health              — liveness + model status
    POST /predict             — score a single transaction
    POST /predict/batch       — score a batch (up to 1 000 transactions)

Notes:
    - flagged_for_review is intentionally excluded from the input schema.
      It is a post-review leakage field and must never be used at inference.
    - Identity (device/session) data is not accepted here because it arrives
      on a separate async feed.  The model handles this gracefully via the
      has_identity=0 / session_count=0 defaults in preprocess().
"""

import pickle
import sys
import warnings
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, validator

# Allow imports from project root when running inside the api/ sub-directory
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from preprocess import preprocess  # noqa: E402

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Umba Fraud Detection API",
    description=(
        "Scores financial transactions and returns a fraud probability "
        "and an alarm decision for the back-office review queue."
    ),
    version="1.0.0",
)

MODEL_PATH = Path(__file__).resolve().parent.parent / "model" / "lgbm_model.pkl"

_artifact: dict = {}   # loaded once at startup


def _load_artifact() -> dict:
    global _artifact
    if not _artifact:
        if not MODEL_PATH.exists():
            raise RuntimeError(
                f"Model not found at {MODEL_PATH}. Run `python train.py` first."
            )
        with open(MODEL_PATH, "rb") as f:
            _artifact = pickle.load(f)
    return _artifact


@app.on_event("startup")
def startup_event():
    _load_artifact()
    print("Model loaded successfully.")


# ---------------------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------------------

class Transaction(BaseModel):
    """Raw transaction fields — mirrors train/test CSV columns."""

    TransactionID:              int
    TransactionDT:              int
    TransactionAmt:             float = Field(..., gt=0, description="Transaction amount (local currency)")
    country:                    str   = Field(..., description="KE or NG")
    currency:                   str   = Field(..., description="KES or NGN")
    channel:                    str   = Field(..., description="mobile_money | p2p | bank_transfer | card | airtime | bill_pay")
    card_type:                  str   = Field(..., description="debit | credit | prepaid")
    card_bank:                  str
    card1:                      Optional[float] = None
    card2:                      Optional[float] = None
    card3:                      Optional[float] = None
    card5:                      Optional[float] = None
    addr1:                      Optional[float] = None
    addr2:                      Optional[float] = None
    dist1:                      Optional[float] = None
    dist2:                      Optional[float] = None
    P_emaildomain:              Optional[str]   = None
    R_emaildomain:              Optional[str]   = None
    recipient_account_age_days: Optional[int]   = None
    sender_prev_txn_count:      Optional[int]   = None
    C1:  Optional[float] = None
    C2:  Optional[float] = None
    C3:  Optional[float] = None
    C4:  Optional[float] = None
    C5:  Optional[float] = None
    C6:  Optional[float] = None
    C7:  Optional[float] = None
    C8:  Optional[float] = None
    D1:  Optional[float] = None
    D2:  Optional[float] = None
    D3:  Optional[float] = None
    D4:  Optional[float] = None
    D5:  Optional[float] = None
    M1:  Optional[str]   = None   # "T" | "F" | null
    M2:  Optional[str]   = None
    M3:  Optional[str]   = None
    M4:  Optional[str]   = None
    M5:  Optional[str]   = None
    M6:  Optional[str]   = None
    V1:  Optional[float] = None
    V2:  Optional[float] = None
    V3:  Optional[float] = None
    V4:  Optional[float] = None
    V5:  Optional[float] = None
    V6:  Optional[float] = None
    V7:  Optional[float] = None
    V8:  Optional[float] = None
    V9:  Optional[float] = None
    V10: Optional[float] = None
    V11: Optional[float] = None
    V12: Optional[float] = None
    V13: Optional[float] = None
    V14: Optional[float] = None
    V15: Optional[float] = None
    V16: Optional[float] = None
    V17: Optional[float] = None
    V18: Optional[float] = None
    V19: Optional[float] = None
    V20: Optional[float] = None

    # flagged_for_review is intentionally absent — post-review leakage field.

    @validator("currency")
    def _validate_currency(cls, v):
        if v not in ("KES", "NGN"):
            raise ValueError("currency must be 'KES' or 'NGN'")
        return v

    @validator("country")
    def _validate_country(cls, v):
        if v not in ("KE", "NG"):
            raise ValueError("country must be 'KE' or 'NG'")
        return v


class PredictionResponse(BaseModel):
    TransactionID:   int
    fraud_prob:      float = Field(..., description="P(fraud) in [0, 1]")
    alarm:           bool  = Field(..., description="True → flag for back-office review")
    alarm_threshold: float
    model_version:   str = "lgbm_v1"


class BatchRequest(BaseModel):
    transactions: List[Transaction] = Field(..., max_items=1000)
    threshold:    float = Field(default=0.5, ge=0.0, le=1.0,
                                description="Override the alarm threshold for this batch")


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

DEFAULT_THRESHOLD = 0.5


def _score(df: pd.DataFrame, feature_cols: list, model) -> np.ndarray:
    """
    Preprocess a transactions dataframe and return fraud probabilities.
    Pads any feature columns absent from the input with NaN so the model
    always receives the full feature matrix it was trained on.
    Numeric columns are explicitly cast to float64; categoricals stay as category.
    """
    df_processed = preprocess(df)
    for col in feature_cols:
        if col not in df_processed.columns:
            df_processed[col] = np.nan

    X = df_processed[feature_cols].copy()

    # Cast non-category columns to float so LightGBM doesn't choke on object dtype
    for col in X.columns:
        if X[col].dtype.name != "category":
            X[col] = pd.to_numeric(X[col], errors="coerce")

    return model.predict(X)


@app.get("/health", summary="Liveness + model check")
def health():
    try:
        art = _load_artifact()
        return {
            "status":      "ok",
            "model_loaded": True,
            "val_pr_auc":  art["metrics"].get("pr_auc"),
            "val_roc_auc": art["metrics"].get("roc_auc"),
        }
    except Exception as exc:
        raise HTTPException(status_code=503, detail=str(exc))


@app.post("/predict", response_model=PredictionResponse, summary="Score a single transaction")
def predict(transaction: Transaction, threshold: float = DEFAULT_THRESHOLD):
    art          = _load_artifact()
    model        = art["model"]
    feature_cols = art["feature_cols"]

    df   = pd.DataFrame([transaction.dict()])
    prob = float(_score(df, feature_cols, model)[0])

    return PredictionResponse(
        TransactionID=transaction.TransactionID,
        fraud_prob=round(prob, 6),
        alarm=prob >= threshold,
        alarm_threshold=threshold,
    )


@app.post("/predict/batch", response_model=List[PredictionResponse],
          summary="Score a batch of transactions (max 1 000)")
def predict_batch(request: BatchRequest):
    if len(request.transactions) > 1000:
        raise HTTPException(status_code=400, detail="Batch size limit is 1 000 transactions.")

    art          = _load_artifact()
    model        = art["model"]
    feature_cols = art["feature_cols"]

    rows  = [t.dict() for t in request.transactions]
    df    = pd.DataFrame(rows)
    probs = _score(df, feature_cols, model)

    return [
        PredictionResponse(
            TransactionID=row["TransactionID"],
            fraud_prob=round(float(prob), 6),
            alarm=float(prob) >= request.threshold,
            alarm_threshold=request.threshold,
        )
        for row, prob in zip(rows, probs)
    ]
