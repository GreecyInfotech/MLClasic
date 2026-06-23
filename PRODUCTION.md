# Production Deployment Guide — All ML Platforms

All seven platforms follow the same production pattern.

## Platforms

| Project | Port | Train Command |
|---------|------|---------------|
| ai-loan-approval-system | 8000 | `python ml/training/train_loan_approval.py` |
| clv-xgboost-lightgbm-platform | 8001 | `python ml/training/train_models.py` |
| customer-segmentation-platform | 8002 | `python ml-platform/training/train_kmeans.py` |
| inventory-forecast-xgboost | 8003 | `python ml/training/train_inventory_forecast.py` |
| next-best-offer-gradient-boosting | 8004 | `python ml/training/train_gradient_boosting.py` |
| container-intelligence-platform | 8005 | `python ml/training/train_dwell_time.py` |
| ml-domain-models-bundle | 8006 | `python ml/training/train_all.py` |

## Standard Production Stack

Each project includes:

- **FastAPI** with lifespan startup, structured logging, CORS, request logging middleware
- **Health probes**: `GET /health` (liveness), `GET /ready` (readiness — 503 until models load)
- **Optional API key auth** via `API_KEY` env + `X-API-Key` header on `/v1/*`
- **PostgreSQL-ready** persistence via `DATABASE_URL` + `psycopg2-binary`
- **Dockerfile** + **docker-compose.yml** (API + Postgres)
- **GitHub Actions CI** (train + pytest)
- **`.env.example`** for configuration

## Deploy Any Platform

```bash
cd <project-directory>
cp .env.example .env
pip install -r requirements.txt
python <train-command-from-table>
docker compose up --build -d
```

## Environment Variables (common)

| Variable | Purpose |
|----------|---------|
| `APP_ENV` | `dev` / `staging` / `prod` |
| `DATABASE_URL` | SQLite (dev) or PostgreSQL (prod) |
| `LOG_LEVEL` | `INFO`, `DEBUG`, `WARNING` |
| `API_KEY` | If set, secures `/v1/*` endpoints |
| `CORS_ORIGINS` | Comma-separated origins or `*` |

## Kubernetes Readiness

```yaml
livenessProbe:
  httpGet:
    path: /health
    port: 8000
readinessProbe:
  httpGet:
    path: /ready
    port: 8000
```

Replace port per platform.

## Security Checklist

- Set strong `API_KEY` in production
- Use PostgreSQL with managed backups
- Run behind TLS-terminating reverse proxy
- Restrict CORS origins in production
- Store secrets in a vault (not committed `.env`)
