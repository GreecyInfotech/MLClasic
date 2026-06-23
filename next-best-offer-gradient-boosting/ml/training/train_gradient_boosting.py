import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split

FEATURE_COLUMNS = [
    "customer_age",
    "income",
    "past_purchases",
    "channel",
    "discount_pct",
    "offer_type",
]


def generate_synthetic_dataset(size: int = 6000) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    customer_age = rng.uniform(22, 70, size=size)
    income = rng.uniform(25000, 180000, size=size)
    past_purchases = rng.uniform(0, 80, size=size)
    channel = rng.integers(0, 4, size=size)
    discount_pct = rng.uniform(0, 40, size=size)
    offer_type = rng.integers(0, 5, size=size)

    score = (
        0.02 * discount_pct
        + 0.001 * past_purchases
        - 0.00001 * income
        + (offer_type == 0) * 0.2
        + rng.normal(0, 0.2, size=size)
    )
    accepted = (score > 0.35).astype(int)

    return pd.DataFrame(
        {
            "customer_age": customer_age,
            "income": income,
            "past_purchases": past_purchases,
            "channel": channel,
            "discount_pct": discount_pct,
            "offer_type": offer_type,
            "accepted_offer": accepted,
        }
    )


def train(df: pd.DataFrame, model_output: Path) -> None:
    X = df[FEATURE_COLUMNS]
    y = df["accepted_offer"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = GradientBoostingClassifier(
        n_estimators=200, learning_rate=0.05, max_depth=4, random_state=42
    )
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    proba = model.predict_proba(X_test)[:, 1]

    model_output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_output)

    metrics = {
        "accuracy": float(accuracy_score(y_test, preds)),
        "roc_auc": float(roc_auc_score(y_test, proba)),
        "train_rows": len(X_train),
        "test_rows": len(X_test),
    }
    (model_output.parent / "training_metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    print(json.dumps(metrics, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-csv", type=Path, default=None)
    parser.add_argument(
        "--model-output", type=Path, default=Path("ml/artifacts/next_best_offer.pkl")
    )
    args = parser.parse_args()

    df = pd.read_csv(args.input_csv) if args.input_csv and args.input_csv.exists() else generate_synthetic_dataset()
    train(df, args.model_output)


if __name__ == "__main__":
    main()
