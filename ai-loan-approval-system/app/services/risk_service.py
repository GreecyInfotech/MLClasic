from app.features import FeatureVector


class RiskService:
    def score(self, features: FeatureVector) -> float:
        score = 0.0

        if features.credit_score < 580:
            score += 0.5
        elif features.credit_score < 650:
            score += 0.25

        if features.debt_to_income > 0.5:
            score += 0.3
        elif features.debt_to_income > 0.35:
            score += 0.15

        if features.loan_to_income > 0.6:
            score += 0.2
        elif features.loan_to_income > 0.4:
            score += 0.1

        if features.delinquencies >= 3:
            score += 0.2

        return min(score, 1.0)
