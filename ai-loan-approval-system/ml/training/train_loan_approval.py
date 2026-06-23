
import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier


FEATURE_COLUMNS = [
    "debt_to_income",
    "loan_to_income",
    "employment_years",
    "credit_score",
    "delinquencies",
]


def generate_synthetic_dataset(size: int = 5000) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    annual_income = rng.integers(30_000, 240_000, size=size)
    requested_amount = rng.integers(5_000, 180_000, size=size)
    existing_debt = rng.integers(0, 160_000, size=size)
    employment_years = rng.uniform(0, 25, size=size)
    credit_score = rng.integers(350, 851, size=size)
    delinquencies = rng.integers(0, 8, size=size)

    debt_to_income = existing_debt / annual_income
    loan_to_income = requested_amount / annual_income

    latent_score = (
        1.3 * (credit_score - 300) / 550
        - 1.8 * debt_to_income
        - 1.0 * loan_to_income
        - 0.18 * delinquencies
        + 0.02 * employment_years
    )
    probability = 1 / (1 + np.exp(-latent_score))
    approved = rng.binomial(1, probability)

    return pd.DataFrame(
        {
            "debt_to_income": debt_to_income,
            "loan_to_income": loan_to_income,
            "employment_years": employment_years,
            "credit_score": credit_score,
            "delinquencies": delinquencies,
            "approved": approved,
        }
    )


def train(df: pd.DataFrame, model_output: Path, metrics_output: Path) -> None:
    X = df[FEATURE_COLUMNS]
    y = df["approved"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = XGBClassifier(
        n_estimators=250,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=42,
        eval_metric="logloss",
    )
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    proba = model.predict_proba(X_test)[:, 1]
    metrics = {
        "accuracy": float(accuracy_score(y_test, preds)),
        "roc_auc": float(roc_auc_score(y_test, proba)),
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "features": FEATURE_COLUMNS,
    }

    model_output.parent.mkdir(parents=True, exist_ok=True)
    metrics_output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_output)
    metrics_output.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"Model saved to {model_output}")
    print(f"Metrics saved to {metrics_output}")
    print(json.dumps(metrics, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train loan approval model")
    parser.add_argument(
        "--input-csv",
        type=Path,
        default=None,
        help="Optional path to a CSV file with training data",
    )
    parser.add_argument(
        "--model-output",
        type=Path,
        default=Path("ml/artifacts/loan_approval_model.pkl"),
        help="Path to write trained model",
    )
    parser.add_argument(
        "--metrics-output",
        type=Path,
        default=Path("ml/artifacts/training_metrics.json"),
        help="Path to write training metrics",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.input_csv and args.input_csv.exists():
        df = pd.read_csv(args.input_csv)
    else:
        df = generate_synthetic_dataset(size=5000)
    train(df, args.model_output, args.metrics_output)


if __name__ == "__main__":
    main()
