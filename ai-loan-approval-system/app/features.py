from dataclasses import dataclass

from app.schemas import LoanApplicationRequest


@dataclass
class FeatureVector:
    debt_to_income: float
    loan_to_income: float
    employment_years: float
    credit_score: int
    delinquencies: int

    def as_model_input(self) -> list[list[float]]:
        return [
            [
                self.debt_to_income,
                self.loan_to_income,
                self.employment_years,
                self.credit_score,
                self.delinquencies,
            ]
        ]


def build_feature_vector(payload: LoanApplicationRequest) -> FeatureVector:
    annual_income = max(payload.annual_income, 1.0)
    debt_to_income = payload.existing_debt / annual_income
    loan_to_income = payload.requested_amount / annual_income
    return FeatureVector(
        debt_to_income=debt_to_income,
        loan_to_income=loan_to_income,
        employment_years=payload.employment_years,
        credit_score=payload.credit_score,
        delinquencies=payload.delinquencies,
    )
