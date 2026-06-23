from dataclasses import dataclass

from app.config import get_settings


@dataclass
class DecisionResult:
    decision: str
    reasons: list[str]


class DecisionEngine:
    def __init__(self) -> None:
        self.settings = get_settings()

    def evaluate(
        self, model_probability: float, fraud_score: float, risk_score: float
    ) -> DecisionResult:
        reasons: list[str] = []

        if fraud_score >= 0.7:
            reasons.append("Fraud score is above rejection threshold.")
            return DecisionResult(decision="rejected", reasons=reasons)

        if model_probability >= self.settings.model_threshold and risk_score < 0.5:
            reasons.append("Model probability and risk profile meet approval criteria.")
            return DecisionResult(decision="approved", reasons=reasons)

        if risk_score >= self.settings.high_risk_threshold:
            reasons.append("Risk score is too high for auto-approval.")
            return DecisionResult(decision="rejected", reasons=reasons)

        reasons.append("Application requires analyst review.")
        return DecisionResult(decision="manual_review", reasons=reasons)
