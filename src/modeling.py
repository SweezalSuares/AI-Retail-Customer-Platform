

import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)

# ------------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------------
BASE = os.path.dirname(__file__)
CLEAN_PATH = os.path.join(BASE, "..", "data", "cleaned_customer_data.csv")
MODEL_DIR = os.path.join(BASE, "..", "outputs", "models")
FIG_DIR = os.path.join(BASE, "..", "outputs", "figures")
REPORT_DIR = os.path.join(BASE, "..", "outputs", "reports")
for d in (MODEL_DIR, FIG_DIR, REPORT_DIR):
    os.makedirs(d, exist_ok=True)

SEGMENT_ORDER = ["Occasional", "Regular", "Loyal", "High_Value"]
RANDOM_STATE = 42

FEATURE_COLS = [
    "age", "annual_income", "months_active", "avg_monthly_spend",
    "purchase_frequency", "avg_order_value", "discount_usage_rate",
    "return_rate", "browsing_time_minutes", "support_interactions",
    "customer_lifetime_value", "spend_to_income_ratio", "engagement_score",
    "loyalty_index", "is_high_returner", "avg_spend_per_order",
    "pay_Card", "pay_UPI", "pay_Wallet",
    "region_Rural", "region_Semi-Urban", "region_Urban",
]

TARGET_COL = "customer_segment_encoded"


def load_data():
    df = pd.read_csv(CLEAN_PATH)
    print(f"[LOAD] {df.shape}")
    cols = [c for c in FEATURE_COLS if c in df.columns]
    missing = set(FEATURE_COLS) - set(cols)
    if missing:
        print(f"[WARN] Missing expected feature columns (skipped): {missing}")
    X = df[cols].to_numpy(dtype=float)   # NumPy feature matrix
    y = df[TARGET_COL].to_numpy(dtype=int)
    return df, X, y, cols


def train_test(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    print(f"[SPLIT] Train: {X_train.shape}, Test: {X_test.shape}")
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler


def evaluate_model(name, model, X_test, y_test):
    y_pred = model.predict(X_test)
    metrics = {
        "model": name,
        "accuracy": round(accuracy_score(y_test, y_pred), 4),
        "precision_macro": round(precision_score(y_test, y_pred, average="macro", zero_division=0), 4),
        "recall_macro": round(recall_score(y_test, y_pred, average="macro", zero_division=0), 4),
        "f1_macro": round(f1_score(y_test, y_pred, average="macro", zero_division=0), 4),
    }
    print(f"[EVAL] {name}: {metrics}")
    return metrics, y_pred


def train_all_models(X_train, X_test, y_train, y_test):
    candidates = {
        "LogisticRegression": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
        "RandomForest": RandomForestClassifier(
            n_estimators=200, max_depth=12, random_state=RANDOM_STATE, n_jobs=-1
        ),
        "GradientBoosting": GradientBoostingClassifier(
            n_estimators=150, max_depth=3, learning_rate=0.1, random_state=RANDOM_STATE
        ),
    }

    results = []
    trained = {}
    predictions = {}
    for name, model in candidates.items():
        model.fit(X_train, y_train)
        metrics, y_pred = evaluate_model(name, model, X_test, y_test)
        results.append(metrics)
        trained[name] = model
        predictions[name] = y_pred

    results_df = pd.DataFrame(results).sort_values("f1_macro", ascending=False)
    best_name = results_df.iloc[0]["model"]
    print(f"\n[BEST MODEL] {best_name}")
    return results_df, trained, predictions, best_name


def plot_confusion_matrix(y_test, y_pred, best_name):
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=SEGMENT_ORDER, yticklabels=SEGMENT_ORDER, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Confusion Matrix — {best_name}")
    fig.tight_layout()
    path = os.path.join(FIG_DIR, "08_confusion_matrix.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[SAVED] {path}")


def plot_model_comparison(results_df):
    fig, ax = plt.subplots(figsize=(8, 5))
    melted = results_df.melt(id_vars="model",
                              value_vars=["accuracy", "precision_macro", "recall_macro", "f1_macro"],
                              var_name="metric", value_name="score")
    sns.barplot(data=melted, x="model", y="score", hue="metric", ax=ax)
    ax.set_title("Model Comparison — Evaluation Metrics")
    ax.set_ylim(0, 1)
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.tight_layout()
    path = os.path.join(FIG_DIR, "09_model_comparison.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[SAVED] {path}")


def plot_feature_importance(model, feature_names, best_name):
    if not hasattr(model, "feature_importances_"):
        print(f"[SKIP] {best_name} has no feature_importances_ attribute")
        return
    importances = model.feature_importances_
    order = np.argsort(importances)[::-1]
    top_n = min(15, len(feature_names))
    fig, ax = plt.subplots(figsize=(9, 7))
    sns.barplot(
        x=importances[order][:top_n],
        y=np.array(feature_names)[order][:top_n],
        color="#385170", ax=ax
    )
    ax.set_title(f"Top {top_n} Feature Importances — {best_name}")
    ax.set_xlabel("Importance")
    fig.tight_layout()
    path = os.path.join(FIG_DIR, "10_feature_importance.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[SAVED] {path}")


def save_artifacts(model, scaler, feature_cols, results_df, best_name, y_test, y_pred):
    joblib.dump(model, os.path.join(MODEL_DIR, "best_model.joblib"))
    joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler.joblib"))
    with open(os.path.join(MODEL_DIR, "feature_columns.json"), "w") as f:
        json.dump(feature_cols, f, indent=2)
    with open(os.path.join(MODEL_DIR, "segment_mapping.json"), "w") as f:
        json.dump({i: s for i, s in enumerate(SEGMENT_ORDER)}, f, indent=2)

    results_df.to_csv(os.path.join(REPORT_DIR, "model_comparison_results.csv"), index=False)

    report_text = classification_report(y_test, y_pred, target_names=SEGMENT_ORDER)
    interpretation = f"""# Modelling Report — Customer Segment Classification

## Problem framing
Predict `customer_segment` (Occasional, Regular, Loyal, High_Value) from
demographic and behavioural features using a supervised multi-class classifier.

## Models compared
{results_df.to_markdown(index=False)}

## Best model: {best_name}

### Classification report (test set)
```
{report_text}
```

## Interpretation
- The best-performing model was selected using macro-averaged F1 score, which
  treats all four segments equally regardless of class imbalance (Occasional
  customers are the majority class in this dataset).
- Feature importance (see outputs/figures/10_feature_importance.png) highlights
  which behavioural signals most strongly separate segments — typically spend,
  purchase frequency, and engagement-derived features dominate over raw
  demographics like age.
- The confusion matrix shows where the model confuses adjacent segments (e.g.
  Regular vs Loyal), which is expected since segment boundaries are behavioural
  gradients rather than hard cutoffs.

## Business use
This model powers the "Predict My Segment" tool in the Streamlit app, allowing
a business user to enter a customer's behavioural profile and get an instant
segment prediction plus the confidence per class — useful for real-time
personalization and targeted retention campaigns.
"""
    with open(os.path.join(REPORT_DIR, "modeling_report.md"), "w") as f:
        f.write(interpretation)
    print(f"[SAVED] Model artifacts to {MODEL_DIR}")
    print(f"[SAVED] Modelling report to {REPORT_DIR}/modeling_report.md")


def run_modeling():
    df, X, y, feature_cols = load_data()
    X_train, X_test, y_train, y_test, scaler = train_test(X, y)
    results_df, trained, predictions, best_name = train_all_models(
        X_train, X_test, y_train, y_test
    )
    best_model = trained[best_name]
    y_pred = predictions[best_name]

    plot_confusion_matrix(y_test, y_pred, best_name)
    plot_model_comparison(results_df)
    plot_feature_importance(best_model, feature_cols, best_name)
    save_artifacts(best_model, scaler, feature_cols, results_df, best_name, y_test, y_pred)

    print("\n[DONE] Modelling complete.")
    return results_df, best_name


if __name__ == "__main__":
    run_modeling()
