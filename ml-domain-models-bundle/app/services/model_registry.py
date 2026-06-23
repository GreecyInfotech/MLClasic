import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from app.config import get_settings
from app.schemas import (
    BerthOccupancyRequest,
    ClaimSettlementRequest,
    CustomerRiskRequest,
    FraudRiskRequest,
    SuspiciousTransactionRequest,
    VesselDelayRequest,
)


class ModelRegistry:
    MODEL_NAMES = [
        "berth-occupancy-lightgbm",
        "claim-settlement-random-forest",
        "customer-risk-scoring-gradient-boosting",
        "fraud-risk-score-gradient-boosting",
        "suspicious-transactions-autoencoder",
        "vessel-delay-prediction-xgboost",
    ]

    def __init__(self) -> None:
        self.settings = get_settings()
        self.models: dict[str, Any] = {}
        self._load_all()

    def _load_all(self) -> None:
        for name in self.MODEL_NAMES:
            path = self.settings.model_path(name)
            if path.exists():
                self.models[name] = joblib.load(path)

    def loaded_models(self) -> list[str]:
        return list(self.models.keys())

    def _require(self, name: str):
        if name not in self.models:
            raise FileNotFoundError(f"Model not loaded: {name}")
        return self.models[name]

    def predict_berth_occupancy(self, payload: BerthOccupancyRequest) -> dict:
        artifact = self._require("berth-occupancy-lightgbm")
        features = np.array(
            [[payload.vessel_count, payload.berth_capacity, payload.hour, payload.day_of_week]]
        )
        occupancy = float(artifact["model"].predict(features)[0])
        return {"occupancy_pct": round(min(max(occupancy, 0), 100), 2)}

    def predict_claim_settlement(self, payload: ClaimSettlementRequest) -> dict:
        artifact = self._require("claim-settlement-random-forest")
        features = np.array(
            [[
                payload.claim_amount,
                payload.policy_age_years,
                payload.prior_claims,
                payload.incident_severity,
            ]]
        )
        approved = int(artifact["model"].predict(features)[0])
        prob = float(artifact["model"].predict_proba(features)[0][1])
        return {"approved": bool(approved), "approval_probability": round(prob, 4)}

    def predict_customer_risk(self, payload: CustomerRiskRequest) -> dict:
        artifact = self._require("customer-risk-scoring-gradient-boosting")
        features = np.array(
            [[
                payload.credit_score,
                payload.debt_ratio,
                payload.delinquencies,
                payload.annual_income,
            ]]
        )
        risk = float(artifact["model"].predict(features)[0])
        return {"risk_score": round(min(max(risk, 0), 1), 4)}

    def predict_fraud_risk(self, payload: FraudRiskRequest) -> dict:
        artifact = self._require("fraud-risk-score-gradient-boosting")
        features = np.array(
            [[
                payload.transaction_amount,
                payload.velocity_24h,
                payload.geo_mismatch,
                payload.merchant_risk,
            ]]
        )
        fraud = int(artifact["model"].predict(features)[0])
        prob = float(artifact["model"].predict_proba(features)[0][1])
        return {"is_fraud": bool(fraud), "fraud_probability": round(prob, 4)}

    def predict_suspicious_transaction(self, payload: SuspiciousTransactionRequest) -> dict:
        artifact = self._require("suspicious-transactions-autoencoder")
        features = np.array(
            [[payload.amount, payload.hour, payload.merchant_category, payload.distance_from_home_km]]
        )
        scaled = artifact["scaler"].transform(features)
        reconstructed = artifact["model"].predict(scaled)
        error = float(np.mean((scaled - reconstructed) ** 2))
        threshold = artifact["threshold"]
        return {
            "anomaly_score": round(error, 6),
            "is_suspicious": error > threshold,
            "threshold": threshold,
        }

    def predict_vessel_delay(self, payload: VesselDelayRequest) -> dict:
        artifact = self._require("vessel-delay-prediction-xgboost")
        features = np.array(
            [[payload.weather_score, payload.port_congestion, payload.cargo_volume_teu]]
        )
        delay = float(artifact["model"].predict(features)[0])
        return {"delay_hours": round(max(delay, 0), 2)}

    def save_audit(self, model_name: str, request: dict, response: dict) -> str:
        return json.dumps({"model": model_name, "request": request, "response": response})
