import json
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor, RandomForestClassifier
from sklearn.metrics import accuracy_score, mean_absolute_error, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler


def _save(model_name: str, artifact: dict, artifacts_dir: Path) -> None:
    out = artifacts_dir / model_name
    out.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, out / "model.pkl")


def train_berth_occupancy(artifacts_dir: Path) -> dict:
    rng = np.random.default_rng(1)
    n = 3000
    vessel_count = rng.uniform(0, 20, n)
    berth_capacity = rng.uniform(5, 25, n)
    hour = rng.integers(0, 24, n)
    day_of_week = rng.integers(0, 7, n)
    occupancy = np.clip(
        (vessel_count / berth_capacity) * 100 + hour * 0.5 + rng.normal(0, 3, n),
        0,
        100,
    )
    df = pd.DataFrame(
        {
            "vessel_count": vessel_count,
            "berth_capacity": berth_capacity,
            "hour": hour,
            "day_of_week": day_of_week,
            "occupancy_pct": occupancy,
        }
    )
    X = df.drop(columns=["occupancy_pct"])
    y = df["occupancy_pct"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = lgb.LGBMRegressor(n_estimators=200, learning_rate=0.05, random_state=42)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    _save("berth-occupancy-lightgbm", {"model": model}, artifacts_dir)
    return {"mae": float(mean_absolute_error(y_test, preds))}


def train_claim_settlement(artifacts_dir: Path) -> dict:
    rng = np.random.default_rng(2)
    n = 4000
    claim_amount = rng.uniform(500, 50000, n)
    policy_age = rng.uniform(0, 20, n)
    prior_claims = rng.integers(0, 6, n)
    severity = rng.integers(1, 6, n)
    approved = ((claim_amount < 20000) & (prior_claims < 3) & (severity < 4)).astype(int)
    df = pd.DataFrame(
        {
            "claim_amount": claim_amount,
            "policy_age_years": policy_age,
            "prior_claims": prior_claims,
            "incident_severity": severity,
            "approved": approved,
        }
    )
    X = df.drop(columns=["approved"])
    y = df["approved"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    model = RandomForestClassifier(n_estimators=200, random_state=42)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    _save("claim-settlement-random-forest", {"model": model}, artifacts_dir)
    return {"accuracy": float(accuracy_score(y_test, preds))}


def train_customer_risk(artifacts_dir: Path) -> dict:
    rng = np.random.default_rng(3)
    n = 4000
    credit = rng.integers(300, 851, n)
    debt_ratio = rng.uniform(0, 1.5, n)
    delinquencies = rng.integers(0, 8, n)
    income = rng.uniform(20000, 200000, n)
    risk = np.clip(
        (850 - credit) / 550 * 0.5 + debt_ratio * 0.3 + delinquencies * 0.05,
        0,
        1,
    )
    df = pd.DataFrame(
        {
            "credit_score": credit,
            "debt_ratio": debt_ratio,
            "delinquencies": delinquencies,
            "annual_income": income,
            "risk_score": risk,
        }
    )
    X = df.drop(columns=["risk_score"])
    y = df["risk_score"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = GradientBoostingRegressor(n_estimators=200, learning_rate=0.05, random_state=42)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    _save("customer-risk-scoring-gradient-boosting", {"model": model}, artifacts_dir)
    return {"mae": float(mean_absolute_error(y_test, preds))}


def train_fraud_risk(artifacts_dir: Path) -> dict:
    rng = np.random.default_rng(4)
    n = 5000
    amount = rng.uniform(10, 10000, n)
    velocity = rng.integers(0, 30, n)
    geo = rng.uniform(0, 1, n)
    merchant = rng.uniform(0, 1, n)
    fraud = ((velocity > 15) | (geo > 0.7) | (merchant > 0.8)).astype(int)
    df = pd.DataFrame(
        {
            "transaction_amount": amount,
            "velocity_24h": velocity,
            "geo_mismatch": geo,
            "merchant_risk": merchant,
            "is_fraud": fraud,
        }
    )
    X = df.drop(columns=["is_fraud"])
    y = df["is_fraud"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    model = GradientBoostingClassifier(n_estimators=200, learning_rate=0.05, random_state=42)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    proba = model.predict_proba(X_test)[:, 1]
    _save("fraud-risk-score-gradient-boosting", {"model": model}, artifacts_dir)
    return {
        "accuracy": float(accuracy_score(y_test, preds)),
        "roc_auc": float(roc_auc_score(y_test, proba)),
    }


def train_suspicious_transactions(artifacts_dir: Path) -> dict:
    rng = np.random.default_rng(5)
    n = 5000
    amount = rng.lognormal(4, 1, n)
    hour = rng.integers(0, 24, n)
    category = rng.integers(0, 21, n)
    distance = rng.uniform(0, 500, n)
    df = pd.DataFrame(
        {"amount": amount, "hour": hour, "merchant_category": category, "distance_from_home_km": distance}
    )
    scaler = StandardScaler()
    X = scaler.fit_transform(df)
    model = MLPRegressor(hidden_layer_sizes=(8, 4, 8), max_iter=500, random_state=42)
    model.fit(X, X)
    recon = model.predict(X)
    errors = np.mean((X - recon) ** 2, axis=1)
    threshold = float(np.quantile(errors, 0.95))
    _save(
        "suspicious-transactions-autoencoder",
        {"model": model, "scaler": scaler, "threshold": threshold},
        artifacts_dir,
    )
    return {"anomaly_threshold": threshold}


def train_vessel_delay(artifacts_dir: Path) -> dict:
    rng = np.random.default_rng(6)
    n = 3500
    weather = rng.uniform(0, 1, n)
    congestion = rng.uniform(0, 1, n)
    cargo = rng.uniform(100, 5000, n)
    delay = weather * 20 + congestion * 30 + cargo / 500 + rng.normal(0, 2, n)
    delay = np.maximum(delay, 0)
    df = pd.DataFrame(
        {
            "weather_score": weather,
            "port_congestion": congestion,
            "cargo_volume_teu": cargo,
            "delay_hours": delay,
        }
    )
    X = df.drop(columns=["delay_hours"])
    y = df["delay_hours"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = xgb.XGBRegressor(n_estimators=250, max_depth=6, learning_rate=0.05, random_state=42)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    _save("vessel-delay-prediction-xgboost", {"model": model}, artifacts_dir)
    return {"mae": float(mean_absolute_error(y_test, preds))}


def main() -> None:
    artifacts_dir = Path("ml/artifacts")
    trainers = {
        "berth-occupancy-lightgbm": train_berth_occupancy,
        "claim-settlement-random-forest": train_claim_settlement,
        "customer-risk-scoring-gradient-boosting": train_customer_risk,
        "fraud-risk-score-gradient-boosting": train_fraud_risk,
        "suspicious-transactions-autoencoder": train_suspicious_transactions,
        "vessel-delay-prediction-xgboost": train_vessel_delay,
    }
    metrics = {}
    for name, trainer in trainers.items():
        print(f"Training {name}...")
        metrics[name] = trainer(artifacts_dir)
    metrics_path = artifacts_dir / "training_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
