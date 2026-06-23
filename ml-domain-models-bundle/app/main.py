import json, logging
from contextlib import asynccontextmanager
from fastapi import Depends, FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from app.config import get_settings
from app.core.exceptions import register_exception_handlers
from app.core.logging import setup_logging
from app.core.middleware import RequestLoggingMiddleware
from app.core.security import verify_api_key
from app.db import Base, engine, get_db
from app.models import PredictionAudit
from app.schemas import BerthOccupancyRequest, ClaimSettlementRequest, CustomerRiskRequest, FraudRiskRequest, PredictionResponse, SuspiciousTransactionRequest, VesselDelayRequest
from app.services.model_registry import ModelRegistry
settings = get_settings(); logger = logging.getLogger(__name__)
registry: ModelRegistry | None = None
@asynccontextmanager
async def lifespan(app: FastAPI):
    global registry
    setup_logging(settings.log_level); Base.metadata.create_all(bind=engine)
    registry = ModelRegistry(); logger.info("Loaded models: %s", registry.loaded_models())
    yield
app = FastAPI(title=settings.app_name, version=settings.app_version, lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=settings.cors_origin_list, allow_credentials=True, allow_methods=["*"], allow_headers=["*"])
app.add_middleware(RequestLoggingMiddleware); register_exception_handlers(app)
@app.get("/health")
def health():
    loaded = registry.loaded_models() if registry else []
    return {"status": "ok", "environment": settings.environment, "loaded_models": loaded}
@app.get("/ready")
def ready():
    if registry is None or len(registry.loaded_models()) < 6:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "Not all models loaded.")
    return {"status": "ready", "loaded_models": registry.loaded_models()}
def _predict_and_audit(model_name, payload, predictor, db):
    if registry is None: raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "Registry not ready.")
    try: result = predictor(payload)
    except FileNotFoundError as exc: raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, str(exc)) from exc
    record = PredictionAudit(model_name=model_name, request_payload=json.dumps(payload), response_payload=json.dumps(result))
    db.add(record); db.commit(); db.refresh(record)
    return PredictionResponse(prediction_id=record.id, model_name=model_name, result=result)
@app.post("/v1/models/berth-occupancy/predict", response_model=PredictionResponse, dependencies=[Depends(verify_api_key)])
def berth_occupancy(payload: BerthOccupancyRequest, db: Session = Depends(get_db)):
    return _predict_and_audit("berth-occupancy-lightgbm", payload.model_dump(), registry.predict_berth_occupancy, db)
@app.post("/v1/models/claim-settlement/predict", response_model=PredictionResponse, dependencies=[Depends(verify_api_key)])
def claim_settlement(payload: ClaimSettlementRequest, db: Session = Depends(get_db)):
    return _predict_and_audit("claim-settlement-random-forest", payload.model_dump(), registry.predict_claim_settlement, db)
@app.post("/v1/models/customer-risk/predict", response_model=PredictionResponse, dependencies=[Depends(verify_api_key)])
def customer_risk(payload: CustomerRiskRequest, db: Session = Depends(get_db)):
    return _predict_and_audit("customer-risk-scoring-gradient-boosting", payload.model_dump(), registry.predict_customer_risk, db)
@app.post("/v1/models/fraud-risk/predict", response_model=PredictionResponse, dependencies=[Depends(verify_api_key)])
def fraud_risk(payload: FraudRiskRequest, db: Session = Depends(get_db)):
    return _predict_and_audit("fraud-risk-score-gradient-boosting", payload.model_dump(), registry.predict_fraud_risk, db)
@app.post("/v1/models/suspicious-transactions/predict", response_model=PredictionResponse, dependencies=[Depends(verify_api_key)])
def suspicious_transactions(payload: SuspiciousTransactionRequest, db: Session = Depends(get_db)):
    return _predict_and_audit("suspicious-transactions-autoencoder", payload.model_dump(), registry.predict_suspicious_transaction, db)
@app.post("/v1/models/vessel-delay/predict", response_model=PredictionResponse, dependencies=[Depends(verify_api_key)])
def vessel_delay(payload: VesselDelayRequest, db: Session = Depends(get_db)):
    return _predict_and_audit("vessel-delay-prediction-xgboost", payload.model_dump(), registry.predict_vessel_delay, db)
@app.get("/v1/predictions/{prediction_id}", dependencies=[Depends(verify_api_key)])
def get_prediction(prediction_id: str, db: Session = Depends(get_db)):
    record = db.query(PredictionAudit).filter(PredictionAudit.id == prediction_id).one_or_none()
    if record is None: raise HTTPException(status.HTTP_404_NOT_FOUND, "Prediction not found.")
    return {"prediction_id": record.id, "model_name": record.model_name, "request": json.loads(record.request_payload), "response": json.loads(record.response_payload), "created_at": record.created_at.isoformat()}
