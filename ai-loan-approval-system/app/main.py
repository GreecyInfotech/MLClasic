import logging
from contextlib import asynccontextmanager
from datetime import datetime

from fastapi import Depends, FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session

from app.config import get_settings
from app.core.exceptions import register_exception_handlers
from app.core.logging import setup_logging
from app.core.middleware import RequestLoggingMiddleware
from app.core.security import verify_api_key
from app.db import Base, engine, get_db
from app.features import build_feature_vector
from app.models import LoanApplication
from app.schemas import LoanApplicationRecord, LoanApplicationRequest, LoanDecisionResponse
from app.services.decision_engine import DecisionEngine
from app.services.fraud_service import FraudService
from app.services.model_service import ModelService
from app.services.risk_service import RiskService

settings = get_settings()
logger = logging.getLogger(__name__)

model_service: ModelService | None = None
fraud_service = FraudService()
risk_service = RiskService()
decision_engine = DecisionEngine()


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model_service
    setup_logging(settings.log_level)
    Base.metadata.create_all(bind=engine)
    try:
        model_service = ModelService()
        logger.info("Loan approval model loaded from %s", settings.model_path)
    except FileNotFoundError:
        model_service = None
        logger.warning("Loan approval model not found at %s", settings.model_path)
    yield


app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origin_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(RequestLoggingMiddleware)
register_exception_handlers(app)


def is_ready() -> bool:
    return model_service is not None


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "environment": settings.environment}


@app.get("/ready")
def ready() -> dict[str, str]:
    if not is_ready():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded.",
        )
    return {"status": "ready"}


@app.post(
    "/v1/loan-applications",
    response_model=LoanDecisionResponse,
    dependencies=[Depends(verify_api_key)],
)
def submit_loan_application(
    payload: LoanApplicationRequest, db: Session = Depends(get_db)
) -> LoanDecisionResponse:
    if model_service is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model is not initialized yet.",
        )

    features = build_feature_vector(payload)
    probability = model_service.predict_probability(features)
    fraud_score = fraud_service.score(payload)
    risk_score = risk_service.score(features)
    decision = decision_engine.evaluate(probability, fraud_score, risk_score)

    record = LoanApplication(
        full_name=payload.full_name,
        email=str(payload.email),
        annual_income=payload.annual_income,
        requested_amount=payload.requested_amount,
        employment_years=payload.employment_years,
        credit_score=payload.credit_score,
        existing_debt=payload.existing_debt,
        delinquencies=payload.delinquencies,
        purpose=payload.purpose.value,
        model_probability=probability,
        fraud_score=fraud_score,
        risk_score=risk_score,
        decision=decision.decision,
        reasons="; ".join(decision.reasons),
    )
    db.add(record)
    db.commit()
    db.refresh(record)

    return LoanDecisionResponse(
        application_id=record.id,
        decision=record.decision,
        model_probability=record.model_probability,
        fraud_score=record.fraud_score,
        risk_score=record.risk_score,
        reasons=decision.reasons,
        timestamp=datetime.utcnow(),
    )


@app.get(
    "/v1/loan-applications/{application_id}",
    response_model=LoanApplicationRecord,
    dependencies=[Depends(verify_api_key)],
)
def get_loan_application(application_id: str, db: Session = Depends(get_db)):
    record = (
        db.query(LoanApplication).filter(LoanApplication.id == application_id).one_or_none()
    )
    if record is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Application {application_id} not found.",
        )
    return record
