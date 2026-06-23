from app.schemas import LoanApplicationRequest


class FraudService:
    def score(self, payload: LoanApplicationRequest) -> float:
        score = 0.05

        if payload.email.endswith(("@mailinator.com", "@tempmail.com")):
            score += 0.4
        if payload.requested_amount > 0.8 * payload.annual_income:
            score += 0.25
        if payload.delinquencies >= 4:
            score += 0.2
        if payload.employment_years < 0.5:
            score += 0.1

        return min(score, 1.0)
