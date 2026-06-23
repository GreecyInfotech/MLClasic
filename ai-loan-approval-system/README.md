AI Loan Approval System

Production-ready end-to-end implementation for loan decisioning:
- Loan approval probability using XGBoost
- Rule-based fraud scoring
- Rule-based risk scoring
- Decision engine (approved/manual review/rejected)
- FastAPI service with validation and persistence
- SQLAlchemy storage (SQLite by default, PostgreSQL-ready via `DATABASE_URL`)

## Project Structure

- `app/main.py` - API entrypoint and endpoints
- `app/services/` - model, fraud, risk, and decision services
- `app/models.py` - persistence model for applications
- `ml/training/train_loan_approval.py` - reproducible model training pipeline
- `tests/test_api.py` - smoke test
- `usermanual.md` - detailed setup and operations guide

## Quick Start

1. Install dependencies:
   - `pip install -r requirements.txt`
2. Train and generate model artifact:
   - `python ml/training/train_loan_approval.py`
3. Run API:
   - `uvicorn app.main:app --reload`
4. Open docs:
   - [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

## Production

```bash
docker compose up --build -d
```

- Liveness: `GET /health`
- Readiness: `GET /ready`
- Optional API key: set `API_KEY` in `.env`

See `usermanual.md` and `../PRODUCTION.md` for full production guide.

## API Endpoints

- `GET /health` - service and model readiness
- `POST /v1/loan-applications` - submit application and get automated decision
- `GET /v1/loan-applications/{application_id}` - fetch stored decision record
