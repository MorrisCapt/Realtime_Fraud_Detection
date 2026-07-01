# Umba Fraud Detection 

Morris Maina | Data Scientist

---

## How to Run Everything

### 1. Prerequisites

- Python 3.11+
- Docker Desktop (for the containerised option)

### 2. Clone and install

```bash
git clone <your-repo-url>
cd Realtime_Fraud_Detection

pip install -r requirements.txt
```

### 3. Place the data

Copy the four CSVs into the `data/` directory:

```
data/
  train.csv
  test.csv
  identity.csv
  sample_submission.csv
```

### 4. Train the model

```bash
python train.py
```

This runs the full pipeline: load → preprocess → time-split → train LightGBM →
evaluate → save. The trained model is written to `model/lgbm_model.pkl` and
validation metrics to `model/val_metrics.json`.

### 5. Score the test set

```bash
python predict.py
```

Outputs `predictions.csv` in the exact format of `sample_submission.csv`.

### 6. Start the API

```bash
uvicorn api.main:app --reload --port 8000
```

- Health check: [http://localhost:8000/health](http://localhost:8000/health)
- Interactive docs: [http://localhost:8000/docs](http://localhost:8000/docs)

#### Example single-transaction request

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "TransactionID": 1120000,
    "TransactionDT": 11657574,
    "TransactionAmt": 5000.0,
    "country": "KE",
    "currency": "KES",
    "channel": "mobile_money",
    "card_type": "debit",
    "card_bank": "mpesa"
  }'
```

Response:

```json
{
  "TransactionID": 1120000,
  "fraud_prob": 0.034,
  "alarm": false,
  "alarm_threshold": 0.5,
  "model_version": "lgbm_v1"
}
```

### 7. Start the dashboard

```bash
streamlit run dashboard/app.py
```

Opens at [http://localhost:8501](http://localhost:8501).

---

## Approach

### Data integrity & leakage checks (done before any modelling)

**`flagged_for_review` is a leakage field and is dropped.**
Fraudulent transactions have a 92.7% flag rate vs 4% for legitimate ones —
making it trivially predictive — but it is populated by the back-office *after*
a human review. It would never be available at the moment we need to score a
transaction, so using it would produce a model that appears perfect in training
but is useless in production. Dropping it is not optional.

**The identity file is not 1-to-1 with transactions.**
Some transactions have two device sessions. Joining naively explodes rows and
introduces duplicate transactions into training. We aggregate per `TransactionID`
first (mean of numeric id_ columns, mode of DeviceType) and add a
`session_count` feature before joining.

**Two currencies on incompatible scales.**
KES amounts average ~2,900 and NGN amounts average ~33,000. We log-transform
`TransactionAmt` to reduce the scale gap, and also keep `currency` as a
categorical feature so the model can learn currency-specific amount thresholds.

**Email domain typos.**
EDA found `gmail.con` (1,899 rows) and `gmial.com` (1,837 rows) as common
misspellings. These are corrected before featurisation.

**Validation must be time-based, not random.**
`test.csv` covers a strictly later period than `train.csv`. A random validation
split would leak future information into training and produce optimistic
estimates. We hold out the last 20% of the training time range as a validation
set — mimicking the production setting.

---

### Feature engineering

| Feature | Rationale |
|---|---|
| `amt_log` | Log-transform of `TransactionAmt` reduces skew and helps the model learn relative, not absolute, amount thresholds |
| `hour_of_day`, `day_of_week` | Fraud is often time-of-day dependent (e.g. late-night bursts) |
| `email_domain_match` | Payer and recipient sharing an email domain is a mild legitimacy signal |
| `recipient_is_new` | Accounts < 30 days old are higher risk |
| `sender_is_new` | First-time senders have no velocity history |
| `has_identity`, `session_count` | Whether device/session data was available, and how many sessions |
| M1–M6 (encoded) | KYC match flags (name, email, ID, device, address, phone) — strong fraud signals |
| C1–C8 | Velocity counts — how many distinct entities have used this card/account |

Anonymised V and D columns are passed through as-is; LightGBM handles their
missing values natively.

---

### Model

**Algorithm: LightGBM (gradient-boosted trees)**

LightGBM is the right tool here because:
- It handles missing values natively (no imputation needed for sparse V/D columns).
- It handles categorical features without one-hot encoding (high-cardinality columns like `card_bank`, email domains).
- It is fast on 120k rows.
- It performs well on tabular fraud data out of the box.

**Imbalance handling: `scale_pos_weight`**

With 3.44% fraud, the dataset has ~28 negatives for every positive.
`scale_pos_weight = 28` re-weights the loss function so rare fraud examples
have equal total weight to the majority class. This is preferable to SMOTE for
tree models: SMOTE synthesises new training points from interpolation but trees
already have mechanisms (leaf constraints, regularisation) to avoid overfitting
on rare classes, and synthetic points can introduce noise near decision boundaries.

**Validation and metric choice: PR-AUC (average precision)**

ROC-AUC is misleading under heavy class imbalance — a model that predicts all
transactions as legitimate scores ~0.5 ROC-AUC but is completely useless. PR-AUC
measures how well the model ranks positives above negatives *and* reflects the
precision-recall trade-off that actually matters operationally.

We also report "operational recall": if ops can only review the top X% riskiest
transactions, what fraction of all fraud do they catch? This directly answers the
question an ops manager would ask.

**Early stopping on val PR-AUC** prevents overfitting without requiring a
separate hyperparameter search.

---

### Threshold selection

The default alarm threshold (0.5) is a starting point. The training pipeline
reports the best-F1 threshold on validation, which balances precision and recall.
In production this decision belongs to the business: a higher threshold means
fewer alarms (ops team reviews less) but more fraud slips through; a lower
threshold catches more fraud but generates more false positives.

The dashboard exposes a threshold slider so the ops manager can see in real time
how the alarm count and catch rate change.

---

### What I'd improve with more time

1. **Hyperparameter tuning** — Optuna TPE search over `num_leaves`, `learning_rate`,
   `min_child_samples`, `reg_alpha/lambda`. Likely +1–2 PR-AUC points.

2. **Calibration** — Apply Platt scaling or isotonic regression so predicted
   probabilities are meaningful as probabilities (not just rankings). This matters
   for threshold selection and for downstream risk systems that multiply
   `fraud_prob × transaction_amount`.

3. **More feature engineering** — Rolling velocity features per card/account over
   different time windows (1h, 24h, 7d) are typically the highest-signal features
   in fraud detection. Not built here because they require a correct time-ordered
   join that is easy to do wrong.

4. **Async identity join in the API** — The real-time API currently scores
   without identity data (it arrives on a separate feed). A production system
   would wait up to ~200 ms for the identity signal, then score.

5. **Production monitoring** — Track score distribution drift weekly using
   Population Stability Index (PSI). Alert when PSI > 0.25. Retrain when
   precision@threshold drops below a business-defined floor on labelled feedback
   from chargebacks/disputes (typically 30–60 day lag).

---

## AI Tool Usage

I used Claude to:
- Draft boilerplate for the FastAPI schema and Pydantic validators.
- Suggest the `aggregate_identity` approach for the one-to-many join.

I reviewed every line and caught/rejected:
- An initial draft that included `flagged_for_review` in the feature set (leakage).
- A random cross-validation setup that didn't respect the time-ordering constraint.
- A suggestion to use SMOTE replaced with `scale_pos_weight` which is better suited to tree-based models.

All data-integrity logic, feature engineering rationale and metric choices are my own.

---

## Project Structure

```
Realtime_Fraud_Detection/
├── data/                   ← place CSVs here
├── model/                  ← model artifact written here after training
├── preprocess.py           ← shared preprocessing (used by train, predict, API)
├── train.py                ← Part A: full pipeline → model/lgbm_model.pkl
├── predict.py              ← Part A: score test.csv → predictions.csv
├── predictions.csv         ← generated by predict.py
├── api/
│   └── main.py             ← Part B: FastAPI service
├── dashboard/
│   └── app.py              ← Part C: Streamlit dashboard
├── Dockerfile.api
├── Dockerfile.dashboard
├── docker-compose.yml      ← Part D: one-command deployment
└── requirements.txt
```
