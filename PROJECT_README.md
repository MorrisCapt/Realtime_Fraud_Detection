# Real-Time Fraud Detection System

A production-ready, end-to-end fraud detection system for mobile-money and banking transactions across **Kenya** 🇰🇪 and **Nigeria** 🇳🇬. Built with LightGBM, FastAPI, Streamlit and Docker.

---

## Table of Contents

1. [Overview](#overview)
2. [The Problem](#the-problem)
3. [Dataset](#dataset)
4. [Project Structure](#project-structure)
5. [How It Works](#how-it-works)
6. [Key Design Decisions](#key-design-decisions)
7. [Model Performance](#model-performance)
8. [Installation & Setup](#installation--setup)
9. [Running Locally](#running-locally)
10. [API Reference](#api-reference)
11. [Dashboard](#dashboard)
12. [Docker Deployment](#docker-deployment)
13. [What I'd Improve With More Time](#what-id-improve-with-more-time)
14. [Tech Stack](#tech-stack)

---

## Overview

When a customer sends money via mobile wallet, bank transfer, or card payment, this system scores the transaction **in real time** and raises an alarm if it looks fraudulent before money moves.

The system consists of three components working together:

```
Transaction comes in
       │
       ▼
  FastAPI Service  ──►  LightGBM Model  ──►  fraud_prob + alarm decision
       │
       ▼
  Streamlit Dashboard  ──►  Ops team reviews flagged transactions
```

---

## The Problem

Financial fraud in mobile money and digital banking is:

- **Rare** — only ~3.4% of transactions are fraudulent, making the dataset heavily imbalanced
- **Time-sensitive** — fraud needs to be caught *before* money moves, not after
- **Evolving** — fraud patterns shift over time, so a model trained on old data must generalise to future transactions
- **Costly** — a missed fraud case (false negative) is far more expensive than a false alarm (false positive)

The goal is to build a model that **ranks transactions by fraud risk** so that an operations team reviewing the top X% of riskiest transactions catches as much fraud as possible.

---

## Dataset

The dataset contains anonymised financial transactions with the following files:

| File | Description |
|---|---|
| `train.csv` | 120,000 labelled transactions used for training |
| `test.csv` | 40,000 unlabelled transactions from a later time period |
| `identity.csv` | Device/session data that joins to transactions on `TransactionID` |
| `sample_submission.csv` | Expected output format |

### Key fields

| Field | Description |
|---|---|
| `TransactionAmt` | Transaction amount in local currency (KES or NGN) |
| `channel` | Payment channel: mobile_money, p2p, bank_transfer, card, airtime, bill_pay |
| `country` | KE (Kenya) or NG (Nigeria) |
| `C1–C8` | Velocity/counting features (how many times a card/account appeared) |
| `M1–M6` | KYC match flags — does name, email, ID, device, address match? |
| `V1–V20` | Anonymised engineered aggregation features |
| `isFraud` | Target label: 1 = fraud, 0 = legitimate (train only) |

### Fraud rate by channel

| Channel | Fraud Rate |
|---|---|
| p2p | 7.1% |
| bank_transfer | 6.1% |
| card | 2.7% |
| mobile_money | 2.3% |
| bill_pay | 1.9% |
| airtime | 1.6% |

---

## Project Structure

```
Realtime_Fraud_Detection/
│
├── data/                        ← Place your CSV files here (not committed)
│   ├── train.csv
│   ├── test.csv
│   ├── identity.csv
│   └── sample_submission.csv
│
├── model/                       ← Trained model artifact (auto-generated)
│   ├── lgbm_model.pkl           ← LightGBM model + feature list + metrics
│   └── val_metrics.json         ← Validation metrics from training
│
├── preprocess.py                ← Shared preprocessing pipeline (train + API)
├── train.py                     ← Full training pipeline → saves model artifact
├── predict.py                   ← Scores test.csv → predictions.csv
├── predictions.csv              ← Model output for all 40,000 test transactions
│
├── api/
│   └── main.py                  ← FastAPI real-time scoring service
│
├── dashboard/
│   └── app.py                   ← Streamlit operations dashboard
│
├── Dockerfile.api               ← Docker image for the API
├── Dockerfile.dashboard         ← Docker image for the dashboard
├── docker-compose.yml           ← Runs both services together
├── requirements.txt             ← Python dependencies
└── README.md                    ← This file
```

---

## How It Works

### 1. Preprocessing (`preprocess.py`)

The same preprocessing pipeline runs identically at training time and inference time — this is critical to avoid **training-serving skew** (where the model sees different data in production than it trained on).

Steps applied:
- **Drop `flagged_for_review`** — this field is populated by a human reviewer *after* a transaction is reviewed. Using it would be leakage since it is never available when we need to score a live transaction.
- **Encode M columns** — KYC match flags (T/F strings) are converted to 1/0 integers.
- **Fix email typos** — `gmail.con` → `gmail.com`, `gmial.com` → `gmail.com` (found in EDA).
- **Engineer new features** — log-transform amount, hour-of-day, day-of-week, email domain match, new account flags.
- **Aggregate identity data** — the identity file has multiple rows per transaction (one per device session). We collapse it to one row per transaction before joining.
- **Cast categoricals** — categorical columns are cast to pandas `category` dtype for LightGBM.

### 2. Training (`train.py`)

```
Load raw CSVs
     │
     ▼
Preprocess (drop leakage, encode, engineer features, join identity)
     │
     ▼
Time-based train/validation split (last 20% of time range = validation)
     │
     ▼
Train LightGBM with scale_pos_weight to handle class imbalance
     │
     ▼
Evaluate on validation set (PR-AUC, ROC-AUC, operational recall)
     │
     ▼
Save model artifact → model/lgbm_model.pkl
```

### 3. Scoring (`predict.py`)

Loads the trained model, preprocesses `test.csv` through the same pipeline, generates a fraud probability for every transaction and saves `predictions.csv`.

### 4. API (`api/main.py`)

A FastAPI service that loads the model once at startup and exposes:
- `GET /health` — liveness check + model status
- `POST /predict` — score a single transaction
- `POST /predict/batch` — score up to 1,000 transactions at once

### 5. Dashboard (`dashboard/app.py`)

A Streamlit app showing the model's behaviour on the test set, with an interactive alarm threshold slider for the operations team.

---

## Key Design Decisions

### Why time-based validation, not random cross-validation?

`test.csv` covers a **later time period** than `train.csv`. This mirrors real production: you train on the past and predict the future. A random split would let future transactions leak into training and produce inflated, unrealistic validation scores. We hold out the last 20% of the training time range as validation to simulate this properly.

### Why PR-AUC and not accuracy or ROC-AUC?

With only 3.4% fraud, a model that labels every transaction as legitimate achieves **96.6% accuracy** — and is completely useless. ROC-AUC has the same problem: it looks good even on heavily imbalanced data.

**PR-AUC (Precision-Recall AUC)** is the honest metric here. It measures how well the model ranks fraud cases above legitimate ones while accounting for the precision-recall tradeoff that actually matters to an operations team.

### Why `scale_pos_weight` for imbalance, not SMOTE?

`scale_pos_weight = 28.9` (ratio of negatives to positives) re-weights the loss function so each fraud case carries 28.9× more weight. This is preferable to SMOTE for tree-based models because:
- SMOTE synthesises new training points from interpolation, which can introduce noise near decision boundaries.
- LightGBM already has mechanisms (leaf constraints, regularisation) to handle rare classes — it just needs the right loss weighting.

### Why LightGBM?

- Handles **missing values natively** — no imputation needed for the sparse V/D columns (up to 72% missing in some).
- Handles **categorical features** without one-hot encoding.
- Fast on 120k rows.
- Consistently strong on tabular fraud data.

### Why drop `flagged_for_review`?

This column has a 92.7% rate among fraud cases and only 4% among legitimate transactions — it appears incredibly predictive. But it is populated by a human reviewer **after** a transaction is flagged for review. It is not available at the moment we need to score a transaction. Including it would produce a model that is perfect in training but useless in production. **It is dropped unconditionally.**

---

## Model Performance

Evaluated on a **time-based holdout** (last 20% of training period — 24,362 transactions, 928 fraud cases):

| Metric | Value |
|---|---|
| PR-AUC (primary) | **0.1784** |
| ROC-AUC | 0.7839 |
| Recall @ top 5% reviewed | **28.3%** (263 of 928 fraud caught) |
| Recall @ top 10% reviewed | **43.1%** (400 of 928 fraud caught) |
| Best F1 threshold | 0.709 |
| Best F1 score | 0.2523 |

### How to read these numbers

**PR-AUC of 0.178** on heavily anonymised data with no velocity features is a solid v1 baseline. A random classifier scores PR-AUC equal to the fraud rate (~0.034), so 0.178 is ~5× better than random.

**Operational recall** is the number that matters most to an ops team: if reviewers can only look at the top 10% of riskiest transactions per day, they will catch **43% of all fraud** — without the model they would catch only 10% by random sampling.

### Top features by importance

The most predictive features (by LightGBM gain) are:
1. `V3`, `V7`, `V17` — anonymised engineered aggregations
2. `M1` — KYC name match flag
3. `channel` — payment channel (p2p and bank_transfer are highest risk)
4. `recipient_account_age_days` — new recipient accounts are higher risk
5. `TransactionAmt` — amount in local currency

---

## Installation & Setup

### Prerequisites
- Python 3.11+
- Docker Desktop (for containerised deployment)

### 1. Clone or unzip the project

```bash
cd Realtime_Fraud_Detection
```

### 2. Create a virtual environment

```bash
# Mac / Linux
python -m venv .venv
source .venv/bin/activate

# Windows
python -m venv .venv
.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Add data files

Place the four CSVs inside the `data/` folder:
```
data/
  train.csv
  test.csv
  identity.csv
  sample_submission.csv
```

---

## Running Locally

### Train the model
```bash
python train.py
```
Outputs: `model/lgbm_model.pkl`, `model/val_metrics.json`

### Generate predictions
```bash
python predict.py
```
Outputs: `predictions.csv` (40,000 rows, one per test transaction)

### Start the API
```bash
uvicorn api.main:app --reload --port 8000
```
- Health check: http://localhost:8000/health
- Interactive docs: http://localhost:8000/docs

### Start the dashboard
```bash
streamlit run dashboard/app.py
```
Opens at: http://localhost:8501

---

## API Reference

### `GET /health`

Returns model status and validation metrics.

```json
{
  "status": "ok",
  "model_loaded": true,
  "val_pr_auc": 0.1784,
  "val_roc_auc": 0.7839
}
```

### `POST /predict`

Score a single transaction.

**Request body:**
```json
{
  "TransactionID": 1120000,
  "TransactionDT": 11657574,
  "TransactionAmt": 5000.0,
  "country": "KE",
  "currency": "KES",
  "channel": "mobile_money",
  "card_type": "debit",
  "card_bank": "mpesa"
}
```

**Response:**
```json
{
  "TransactionID": 1120000,
  "fraud_prob": 0.034,
  "alarm": false,
  "alarm_threshold": 0.5,
  "model_version": "lgbm_v1"
}
```

### `POST /predict/batch`

Score up to 1,000 transactions in one call.

```json
{
  "transactions": [ { ... }, { ... } ],
  "threshold": 0.5
}
```

All optional fields (C1–C8, M1–M6, V1–V20, D1–D5, etc.) can be omitted — the model handles missing values natively.

---

## Dashboard

The Streamlit dashboard gives the operations team a real-time view of model behaviour:

- **KPI cards** — total transactions, flagged count, mean fraud score, PR-AUC
- **Score distribution** — histogram of fraud probabilities with adjustable alarm threshold
- **Flag rate by channel** — which payment channels are most at risk
- **Country and card type breakdown** — fraud rates by market
- **Top 100 flagged transactions** — sortable table with amounts, channels, and scores
- **Validation metrics** — PR-AUC, ROC-AUC, recall at top 5% and 10% reviewed

The threshold slider lets ops managers trade off precision (fewer false alarms) vs recall (catching more fraud) based on their team's daily review capacity.

---

## Docker Deployment

Run the API and dashboard together with a single command:

```bash
docker compose up --build
```

| Service | URL |
|---|---|
| API | http://localhost:8000 |
| API docs | http://localhost:8000/docs |
| Dashboard | http://localhost:8501 |

Stop everything:
```bash
docker compose down
```

Rebuild after code changes:
```bash
docker compose up --build
```

After retraining the model (no code changes needed):
```bash
docker compose restart
```

---

## What I'd Improve With More Time

### 1. Velocity features
Rolling counts per card/account over 1h, 24h, and 7d windows are typically the single highest-signal feature family in fraud detection. They require a careful time-ordered join to avoid leakage and were left out of v1.

### 2. Hyperparameter tuning
Optuna TPE search over `num_leaves`, `learning_rate`, `min_child_samples`, `reg_alpha/lambda`. Estimated +1–2 PR-AUC points.

### 3. Probability calibration
Apply Platt scaling or isotonic regression so `fraud_prob` is a true probability (not just a ranking score). This matters for downstream systems that multiply `fraud_prob × transaction_amount` to prioritise review.

### 4. Production monitoring
- Track score distribution weekly using Population Stability Index (PSI). Alert when PSI > 0.25.
- Track precision@threshold on labelled feedback (chargebacks/disputes arrive with 30–60 day lag).
- Retrain when precision drops below a business-defined floor.

### 5. Async identity enrichment in the API
The current API scores without device/session data since that arrives on a separate feed. A production system would wait up to ~200ms for the identity signal before scoring.

---

## Tech Stack

| Layer | Technology |
|---|---|
| Language | Python 3.11 |
| ML model | LightGBM 4.0 |
| Data processing | pandas, numpy |
| API | FastAPI + Uvicorn |
| Dashboard | Streamlit + Matplotlib |
| Containerisation | Docker + Docker Compose |

---

*Built by Morris Maina*
