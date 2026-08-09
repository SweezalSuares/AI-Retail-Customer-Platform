
import pandas as pd
import numpy as np
import os

# ------------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------------
RAW_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "retail_customer_segmentation.csv")
CLEAN_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "cleaned_customer_data.csv")

NUMERIC_COLS_TO_IMPUTE = [
    "annual_income", "avg_monthly_spend", "purchase_frequency",
    "discount_usage_rate", "return_rate", "browsing_time_minutes",
    "support_interactions"
]

# Columns that are naturally skewed and benefit from IQR-based outlier capping
SKEWED_COLS = [
    "annual_income", "avg_monthly_spend", "avg_order_value",
    "browsing_time_minutes", "purchase_frequency"
]


def load_data(path: str) -> pd.DataFrame:
    """Load the raw CSV into a DataFrame."""
    df = pd.read_csv(path)
    print(f"[LOAD] Raw shape: {df.shape}")
    return df


def structural_checks(df: pd.DataFrame) -> pd.DataFrame:
    """Basic sanity checks: duplicates, id integrity, dtype summary."""
    n_dupes = df.duplicated(subset="customer_id").sum()
    if n_dupes > 0:
        print(f"[CLEAN] Dropping {n_dupes} duplicate customer_id rows")
        df = df.drop_duplicates(subset="customer_id", keep="first")

    # Drop fully-empty rows if any
    before = len(df)
    df = df.dropna(how="all")
    if len(df) != before:
        print(f"[CLEAN] Dropped {before - len(df)} fully empty rows")

    return df.reset_index(drop=True)


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Impute missing numeric values using median (robust to outliers/skew).
    We use median rather than mean because several columns (income, spend,
    browsing time) are right-skewed, so the mean would be pulled upward by
    a small number of very high-value customers.
    """
    missing_before = df[NUMERIC_COLS_TO_IMPUTE].isnull().sum()
    print("[MISSING] Missing values before imputation:")
    print(missing_before[missing_before > 0])

    for col in NUMERIC_COLS_TO_IMPUTE:
        if col in df.columns:
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)

    # support_interactions is count-like -> round after imputation
    if "support_interactions" in df.columns:
        df["support_interactions"] = df["support_interactions"].round().astype(int)

    assert df[NUMERIC_COLS_TO_IMPUTE].isnull().sum().sum() == 0, "Missing values remain!"
    print("[MISSING] All missing values imputed.")
    return df


def cap_outliers_iqr(df: pd.DataFrame, cols: list, k: float = 1.5) -> pd.DataFrame:
    """Cap outliers to [Q1 - k*IQR, Q3 + k*IQR] for the given numeric columns."""
    for col in cols:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - k * iqr
        upper = q3 + k * iqr
        n_capped = ((df[col] < lower) | (df[col] > upper)).sum()
        df[col] = df[col].clip(lower=lower, upper=upper)
        if n_capped > 0:
            print(f"[OUTLIERS] {col}: capped {n_capped} values to [{lower:.2f}, {upper:.2f}]")
    return df


def clean_basic_validity(df: pd.DataFrame) -> pd.DataFrame:
    """Enforce logical constraints (rates between 0-1, non-negative values)."""
    for col in ["discount_usage_rate", "return_rate"]:
        df[col] = df[col].clip(lower=0, upper=1)

    for col in ["age", "months_active", "annual_income", "avg_monthly_spend",
                "avg_order_value", "browsing_time_minutes", "support_interactions"]:
        df[col] = df[col].clip(lower=0)

    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create new, business-meaningful features for EDA and modelling.

    - customer_lifetime_value (CLV proxy) = avg_monthly_spend * months_active
    - spend_to_income_ratio   = how much of their income a customer spends
    - engagement_score        = normalized blend of browsing time & purchase frequency
    - loyalty_index           = months_active weighted by purchase_frequency
    - is_high_returner        = flag for return_rate above 75th percentile
    - avg_spend_per_order     = avg_monthly_spend / purchase_frequency (guard div/0)
    """
    df["customer_lifetime_value"] = df["avg_monthly_spend"] * df["months_active"]

    df["spend_to_income_ratio"] = (
        (df["avg_monthly_spend"] * 12) / df["annual_income"].replace(0, np.nan)
    ).fillna(0).round(4)

    # min-max normalize browsing time & purchase frequency, then average
    bt_norm = (df["browsing_time_minutes"] - df["browsing_time_minutes"].min()) / (
        df["browsing_time_minutes"].max() - df["browsing_time_minutes"].min()
    )
    pf_norm = (df["purchase_frequency"] - df["purchase_frequency"].min()) / (
        df["purchase_frequency"].max() - df["purchase_frequency"].min()
    )
    df["engagement_score"] = ((bt_norm + pf_norm) / 2).round(4)

    df["loyalty_index"] = (df["months_active"] * df["purchase_frequency"]).round(2)

    return_75th = df["return_rate"].quantile(0.75)
    df["is_high_returner"] = (df["return_rate"] >= return_75th).astype(int)

    df["avg_spend_per_order"] = (
        df["avg_monthly_spend"] / df["purchase_frequency"].replace(0, np.nan)
    ).fillna(0).round(2)

    # Age bracket for segment-friendly grouping
    df["age_group"] = pd.cut(
        df["age"], bins=[17, 25, 35, 45, 55, 70],
        labels=["18-25", "26-35", "36-45", "46-55", "56-70"]
    )

    print("[FEATURES] Added: customer_lifetime_value, spend_to_income_ratio, "
          "engagement_score, loyalty_index, is_high_returner, avg_spend_per_order, age_group")
    return df


def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Label-encode the target (customer_segment) into an ordinal-friendly mapping
    and one-hot encode nominal categorical predictors (payment_method, region).
    age_group is kept as a category for EDA but also one-hot encoded for modelling.
    """
    segment_order = {"Occasional": 0, "Regular": 1, "Loyal": 2, "High_Value": 3}
    df["customer_segment_encoded"] = df["customer_segment"].map(segment_order)

    # Keep original categorical columns (needed for EDA / Streamlit filters)
    # while ALSO adding one-hot encoded versions (needed for modelling).
    dummies = pd.get_dummies(
        df[["payment_method", "region", "age_group"]],
        prefix=["pay", "region", "age"], drop_first=False
    )
    df = pd.concat([df, dummies], axis=1)

    return df


def run_pipeline():
    df = load_data(RAW_PATH)
    df = structural_checks(df)
    df = handle_missing_values(df)
    df = clean_basic_validity(df)
    df = cap_outliers_iqr(df, SKEWED_COLS)
    df = engineer_features(df)
    df = encode_categoricals(df)

    df.to_csv(CLEAN_PATH, index=False)
    print(f"\n[DONE] Cleaned dataset saved to: {CLEAN_PATH}")
    print(f"[DONE] Final shape: {df.shape}")
    return df


if __name__ == "__main__":
    run_pipeline()
