# User Manual - AI Loan Approval System

## 1) Overview

The system provides an end-to-end automated loan decisioning flow:
1. Train an ML model (`XGBoost`) for approval probability.
2. Accept loan applications through a REST API.
3. Compute fraud score and risk score.
4. Combine ML + rules in a decision engine.
5. Persist all decisions for audit and retrieval.

Decision outputs:
- `approved`
- `manual_review`
- `rejected`

## 2) Prerequisites

- Python 3.10+
- `pip`

Optional for production:
- PostgreSQL 14+ (instead of default SQLite)
- Linux container runtime (Docker/Kubernetes)

## 3) Installation

From repository root:

```bash
pip install -r requirements.txt
```

Copy env template:

```bash
copy .env.example .env
```

(On Unix use `cp .env.example .env`)

## 4) Configuration

Set values in `.env`:

- `APP_ENV`: `dev`, `staging`, or `prod`
- `DATABASE_URL`: SQLAlchemy URL
  - default: `sqlite:///./loan_approval.db`
  - postgres example: `postgresql+psycopg2://user:password@localhost:5432/loan_db`
- `MODEL_PATH`: model artifact path
- `MODEL_THRESHOLD`: minimum ML probability for auto-approval
- `HIGH_RISK_THRESHOLD`: risk cutoff for rejection

## 5) Model Training

### Option A: Synthetic data (default)

```bash
python ml/training/train_loan_approval.py
```

Generated artifacts:
- `ml/artifacts/loan_approval_model.pkl`
- `ml/artifacts/training_metrics.json`

### Option B: Custom CSV input

CSV must contain:
- `debt_to_income`
- `loan_to_income`
- `employment_years`
- `credit_score`
- `delinquencies`
- `approved` (0/1 target)

Run:

```bash
python ml/training/train_loan_approval.py --input-csv path/to/loan_data.csv
```

## 6) Run the API

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Interactive docs:
- [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

Health check:

```bash
curl http://127.0.0.1:8000/health
```

- `{"status":"ok"}` means model loaded.
- `{"status":"degraded"}` means service is up but model artifact is missing.

## 7) End-to-End Loan Flow

### Step 1: Submit loan application

```bash
curl -X POST "http://127.0.0.1:8000/v1/loan-applications" ^
  -H "Content-Type: application/json" ^
  -d "{\"full_name\":\"Jane Doe\",\"email\":\"jane@example.com\",\"annual_income\":95000,\"requested_amount\":18000,\"employment_years\":6,\"credit_score\":720,\"existing_debt\":12000,\"delinquencies\":0,\"purpose\":\"personal\"}"
```

Example response:

```json
{
  "application_id": "f7c42f69-9ad6-4afb-bafe-644bbf45f4ea",
  "decision": "approved",
  "model_probability": 0.84,
  "fraud_score": 0.05,
  "risk_score": 0.15,
  "reasons": ["Model probability and risk profile meet approval criteria."],
  "timestamp": "2026-06-23T09:35:20.123456"
}
```

### Step 2: Retrieve application audit record

```bash
curl "http://127.0.0.1:8000/v1/loan-applications/{application_id}"
```

Returned record includes full request inputs + computed scores + final decision.

## 8) Decision Logic

1. **Fraud gate first**
   - If fraud score `>= 0.7` -> `rejected`
2. **Auto approval**
   - If model probability `>= MODEL_THRESHOLD` and risk score `< 0.5` -> `approved`
3. **Risk rejection**
   - If risk score `>= HIGH_RISK_THRESHOLD` -> `rejected`
4. **Fallback**
   - Otherwise -> `manual_review`

## 9) Testing

Run:

```bash
pytest -q
```

## Production Deployment

### Docker

```bash
docker compose up --build -d
```

### Health Probes

- Liveness: `GET /health` (always returns 200 when process is up)
- Readiness: `GET /ready` (returns 503 until model artifact is loaded)

### Security

Set `API_KEY` in `.env` to require `X-API-Key` header on all `/v1/*` endpoints.

### PostgreSQL

For production, set `DATABASE_URL` to a PostgreSQL connection string (see `docker-compose.yml`).

- Use PostgreSQL instead of SQLite.
- Put API behind a reverse proxy/load balancer.
- Store `.env` values in a secrets manager.
- Add authentication/authorization for API endpoints.
- Add structured logging and telemetry (OpenTelemetry).
- Introduce idempotency key for request replay safety.
- Add async queue/event streaming for manual-review workflow.
- Add model monitoring and drift alerts for `training_metrics.json`.

## 11) Troubleshooting

- **`503 Model is not initialized yet`**
  - Train model and ensure `MODEL_PATH` points to valid `.pkl`.
- **`422 Unprocessable Entity`**
  - Verify request fields and ranges (credit score 300-850, positive amounts).
- **DB connection errors**
  - Confirm `DATABASE_URL` format and DB reachability.
