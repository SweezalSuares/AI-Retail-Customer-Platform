import os
import json
import numpy as np
import pandas as pd
import joblib

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, IsolationForest
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, r2_score

BASE = os.path.dirname(__file__)
CLEAN_PATH = os.path.join(BASE, "..", "data", "cleaned_customer_data.csv")
MODEL_DIR = os.path.join(BASE, "..", "outputs", "models")
REPORT_DIR = os.path.join(BASE, "..", "outputs", "reports")
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)

RANDOM_STATE = 42

BEHAVIOUR_FEATURES = [
    "age", "annual_income", "months_active", "avg_monthly_spend",
    "purchase_frequency", "avg_order_value", "discount_usage_rate",
    "return_rate", "browsing_time_minutes", "support_interactions",
    "engagement_score", "loyalty_index",
]


# ------------------------------------------------------------------
# 1. CHURN RISK
# ------------------------------------------------------------------
def build_churn_proxy_label(df: pd.DataFrame) -> pd.Series:
    """
    Documented churn-risk proxy (0=Low, 1=Medium, 2=High) built from:
      - purchase_frequency  (lower  -> higher risk)
      - engagement_score    (lower  -> higher risk)
      - months_active       (shorter tenure + low frequency -> higher risk)
      - return_rate         (very high -> mild risk signal, dissatisfaction)

    Each driver is min-max normalized and combined into a single risk score;
    the score is then bucketed into tertiles (Low/Medium/High). This is a
    heuristic label used because no explicit churn event exists in the data.
    """
    freq_n = 1 - _minmax(df["purchase_frequency"])
    eng_n = 1 - _minmax(df["engagement_score"])
    tenure_n = 1 - _minmax(df["months_active"])
    return_n = _minmax(df["return_rate"]) * 0.5  # smaller weight, secondary signal

    risk_score = (freq_n * 0.35 + eng_n * 0.35 + tenure_n * 0.20 + return_n * 0.10)
    buckets = pd.qcut(risk_score, q=3, labels=[0, 1, 2])  # 0=Low,1=Medium,2=High
    return risk_score.round(4), buckets.astype(int)


def _minmax(s: pd.Series) -> pd.Series:
    return (s - s.min()) / (s.max() - s.min() + 1e-9)


def train_churn_model(df: pd.DataFrame):
    risk_score, risk_bucket = build_churn_proxy_label(df)
    df = df.copy()
    df["churn_risk_score"] = risk_score
    df["churn_risk_bucket"] = risk_bucket

    X = df[BEHAVIOUR_FEATURES].to_numpy(dtype=float)
    y = risk_bucket.to_numpy(dtype=int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    clf = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=RANDOM_STATE, n_jobs=-1)
    clf.fit(X_train_s, y_train)
    y_pred = clf.predict(X_test_s)

    metrics = {
        "accuracy": round(accuracy_score(y_test, y_pred), 4),
        "f1_macro": round(f1_score(y_test, y_pred, average="macro"), 4),
    }
    print(f"[CHURN] {metrics}")

    joblib.dump(clf, os.path.join(MODEL_DIR, "churn_model.joblib"))
    joblib.dump(scaler, os.path.join(MODEL_DIR, "churn_scaler.joblib"))
    with open(os.path.join(MODEL_DIR, "churn_features.json"), "w") as f:
        json.dump(BEHAVIOUR_FEATURES, f, indent=2)

    return clf, scaler, metrics, df


# ------------------------------------------------------------------
# 2. CLV FORECAST
# ------------------------------------------------------------------
def train_clv_forecast_model(df: pd.DataFrame):
    """
    Predict avg_monthly_spend from non-circular behavioural/demographic
    drivers, then project forward using an engagement-weighted expected
    additional tenure. Forecast CLV = predicted_monthly_spend * expected_months.
    """
    drivers = [
        "age", "annual_income", "purchase_frequency", "avg_order_value",
        "discount_usage_rate", "return_rate", "browsing_time_minutes",
        "support_interactions", "engagement_score",
    ]
    X = df[drivers].to_numpy(dtype=float)
    y = df["avg_monthly_spend"].to_numpy(dtype=float)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )
    reg = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=RANDOM_STATE, n_jobs=-1)
    reg.fit(X_train, y_train)
    y_pred = reg.predict(X_test)

    metrics = {
        "mae": round(mean_absolute_error(y_test, y_pred), 2),
        "r2": round(r2_score(y_test, y_pred), 4),
    }
    print(f"[CLV] {metrics}")

    joblib.dump(reg, os.path.join(MODEL_DIR, "clv_forecast_model.joblib"))
    with open(os.path.join(MODEL_DIR, "clv_drivers.json"), "w") as f:
        json.dump(drivers, f, indent=2)

    return reg, drivers, metrics


def project_future_clv(predicted_monthly_spend: float, months_active: float, engagement_score: float,
                        horizon_months: int = 12) -> float:
    """
    Expected additional tenure grows with engagement (a proxy for retention
    likelihood). This is an illustrative estimation formula, not a guarantee.
    """
    expected_additional_months = min(horizon_months, 3 + engagement_score * horizon_months)
    return round(predicted_monthly_spend * expected_additional_months, 2)


# ------------------------------------------------------------------
# 3. EXPLAINABLE AI (local contribution breakdown)
# ------------------------------------------------------------------
def explain_prediction(feature_values: dict, feature_names: list, importances: np.ndarray,
                        reference_means: dict) -> pd.DataFrame:
    """
    Simple, transparent local explanation: for each feature, how far the
    customer's value sits from the dataset average (z-style deviation),
    weighted by that feature's global importance in the model. Positive
    contribution = pushes prediction toward higher-value/higher-risk class;
    magnitude indicates influence, not exact SHAP-level precision.
    """
    rows = []
    for i, feat in enumerate(feature_names):
        val = feature_values.get(feat, 0)
        mean = reference_means.get(feat, 0)
        std = reference_means.get(f"{feat}_std", 1) or 1
        deviation = (val - mean) / std
        contribution = deviation * importances[i]
        rows.append({"feature": feat, "value": val, "importance": round(importances[i], 4),
                      "deviation_from_avg": round(deviation, 2), "contribution": round(contribution, 4)})
    exp_df = pd.DataFrame(rows).sort_values("contribution", key=abs, ascending=False)
    return exp_df


# ------------------------------------------------------------------
# 4. BEHAVIOURAL CLUSTERING (unsupervised AI segments)
# ------------------------------------------------------------------
CLUSTER_LABELS_TEMPLATE = {
    # filled dynamically based on cluster centroid characteristics
}


def train_behavioural_clusters(df: pd.DataFrame, k: int = 5):
    X = df[BEHAVIOUR_FEATURES].to_numpy(dtype=float)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
    clusters = km.fit_predict(X_scaled)
    df = df.copy()
    df["ai_cluster"] = clusters

    profiles = df.groupby("ai_cluster")[BEHAVIOUR_FEATURES].mean().round(2)
    labels = _auto_label_clusters(profiles)
    df["ai_cluster_label"] = df["ai_cluster"].map(labels)

    joblib.dump(km, os.path.join(MODEL_DIR, "cluster_model.joblib"))
    joblib.dump(scaler, os.path.join(MODEL_DIR, "cluster_scaler.joblib"))
    profiles["label"] = profiles.index.map(labels)
    profiles.to_csv(os.path.join(REPORT_DIR, "cluster_profiles.csv"))
    with open(os.path.join(MODEL_DIR, "cluster_labels.json"), "w") as f:
        json.dump({str(k_): v for k_, v in labels.items()}, f, indent=2)

    print(f"[CLUSTERING] {k} behavioural clusters trained")
    print(profiles[["label"]])
    return df, profiles, labels


def _auto_label_clusters(profiles: pd.DataFrame) -> dict:
    """Heuristically name each cluster from its centroid characteristics, ranked
    by distinguishing traits so that clusters don't collide on the same label."""
    overall = profiles.mean()
    scored = {}
    for idx, row in profiles.iterrows():
        spend_z = row["avg_monthly_spend"] - overall["avg_monthly_spend"]
        freq_z = row["purchase_frequency"] - overall["purchase_frequency"]
        income_z = row["annual_income"] - overall["annual_income"]
        eng_z = row["engagement_score"] - overall["engagement_score"]
        return_z = row["return_rate"] - overall["return_rate"]
        discount_z = row["discount_usage_rate"] - overall["discount_usage_rate"]
        scored[idx] = {
            "Premium Customers": spend_z + income_z + freq_z,
            "Bargain Hunters": discount_z - spend_z,
            "Frequent Small-Basket Buyers": freq_z - spend_z,
            "At-Risk / Dormant": -(eng_z + freq_z),
            "High-Return Shoppers": return_z,
            "Emerging Customers": -abs(income_z) - abs(spend_z),  # closest to average
        }

    labels = {}
    used = set()
    # Assign each cluster its single strongest-scoring label, avoiding repeats
    remaining = dict(scored)
    while remaining:
        # find the (cluster, label) pair with the highest score globally
        best_cluster, best_label, best_score = None, None, -np.inf
        for c, label_scores in remaining.items():
            for label, score in label_scores.items():
                if label in used:
                    continue
                if score > best_score:
                    best_cluster, best_label, best_score = c, label, score
        labels[best_cluster] = best_label
        used.add(best_label)
        del remaining[best_cluster]

    return labels


# ------------------------------------------------------------------
# 5. ANOMALY / FRAUD-PROXY DETECTION
# ------------------------------------------------------------------
def train_anomaly_detector(df: pd.DataFrame, contamination: float = 0.03):
    """
    IsolationForest flags statistically unusual behavioural combinations
    (e.g. very high return_rate + very high discount_usage_rate + low
    support_interactions) as anomalies worth manual review. There is no
    ground-truth fraud label in this dataset, so treat output as a
    prioritization signal, not a verdict.
    """
    feats = ["avg_monthly_spend", "avg_order_value", "discount_usage_rate",
              "return_rate", "support_interactions", "purchase_frequency"]
    X = df[feats].to_numpy(dtype=float)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    iso = IsolationForest(contamination=contamination, random_state=RANDOM_STATE, n_estimators=200)
    preds = iso.fit_predict(X_scaled)   # -1 = anomaly, 1 = normal
    scores = iso.decision_function(X_scaled)  # higher = more normal

    df = df.copy()
    df["anomaly_flag"] = (preds == -1).astype(int)
    df["anomaly_score"] = (-scores).round(4)  # flip so higher = more anomalous

    joblib.dump(iso, os.path.join(MODEL_DIR, "anomaly_model.joblib"))
    joblib.dump(scaler, os.path.join(MODEL_DIR, "anomaly_scaler.joblib"))
    with open(os.path.join(MODEL_DIR, "anomaly_features.json"), "w") as f:
        json.dump(feats, f, indent=2)

    n_flagged = df["anomaly_flag"].sum()
    print(f"[ANOMALY] Flagged {n_flagged} customers ({n_flagged/len(df)*100:.1f}%) as unusual")
    return df, feats


# ------------------------------------------------------------------
# ORCHESTRATOR
# ------------------------------------------------------------------
def run_advanced_analytics():
    df = pd.read_csv(CLEAN_PATH)
    print(f"[LOAD] {df.shape}")

    churn_model, churn_scaler, churn_metrics, df_churn = train_churn_model(df)
    clv_model, clv_drivers, clv_metrics = train_clv_forecast_model(df)
    df_clustered, profiles, cluster_labels = train_behavioural_clusters(df, k=5)
    df_anomaly, anomaly_feats = train_anomaly_detector(df)

    # Merge derived columns back into one enriched dataset for the app
    enriched = df.copy()
    enriched["churn_risk_score"] = df_churn["churn_risk_score"]
    enriched["churn_risk_bucket"] = df_churn["churn_risk_bucket"]
    enriched["ai_cluster"] = df_clustered["ai_cluster"]
    enriched["ai_cluster_label"] = df_clustered["ai_cluster_label"]
    enriched["anomaly_flag"] = df_anomaly["anomaly_flag"]
    enriched["anomaly_score"] = df_anomaly["anomaly_score"]

    enriched_path = os.path.join(BASE, "..", "data", "enriched_customer_data.csv")
    enriched.to_csv(enriched_path, index=False)
    print(f"[SAVED] {enriched_path}")

    summary = {
        "churn_model_metrics": churn_metrics,
        "clv_forecast_metrics": clv_metrics,
        "n_clusters": 5,
        "cluster_labels": {str(k): v for k, v in cluster_labels.items()},
        "n_anomalies_flagged": int(enriched["anomaly_flag"].sum()),
    }
    with open(os.path.join(REPORT_DIR, "advanced_analytics_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print("\n[DONE] Advanced analytics complete.")
    return summary


if __name__ == "__main__":
    run_advanced_analytics()
