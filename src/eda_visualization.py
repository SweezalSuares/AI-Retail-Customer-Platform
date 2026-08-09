
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import os

# ------------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------------
BASE = os.path.dirname(__file__)
CLEAN_PATH = os.path.join(BASE, "..", "data", "cleaned_customer_data.csv")
FIG_DIR = os.path.join(BASE, "..", "outputs", "figures")
REPORT_DIR = os.path.join(BASE, "..", "outputs", "reports")
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)

sns.set_theme(style="whitegrid", palette="viridis")
SEGMENT_ORDER = ["Occasional", "Regular", "Loyal", "High_Value"]
SEGMENT_PALETTE = {"Occasional": "#9fd3c7", "Regular": "#385170",
                    "Loyal": "#f38181", "High_Value": "#fce38a"}


def load_clean(path=CLEAN_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    print(f"[LOAD] Cleaned shape: {df.shape}")
    return df


def save_fig(fig, name):
    path = os.path.join(FIG_DIR, name)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[SAVED] {path}")


# ------------------------------------------------------------------
# 1. Segment distribution
# ------------------------------------------------------------------
def plot_segment_distribution(df):
    fig, ax = plt.subplots(figsize=(7, 5))
    counts = df["customer_segment"].value_counts().reindex(SEGMENT_ORDER)
    sns.barplot(x=counts.index, y=counts.values,
                palette=[SEGMENT_PALETTE[s] for s in counts.index], ax=ax)
    ax.set_title("Customer Segment Distribution")
    ax.set_xlabel("Segment")
    ax.set_ylabel("Number of Customers")
    for i, v in enumerate(counts.values):
        ax.text(i, v + 200, f"{v:,}", ha="center", fontweight="bold")
    save_fig(fig, "01_segment_distribution.png")


# ------------------------------------------------------------------
# 2. Numeric distributions
# ------------------------------------------------------------------
def plot_numeric_distributions(df):
    cols = ["age", "annual_income", "avg_monthly_spend", "purchase_frequency",
            "avg_order_value", "browsing_time_minutes"]
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    axes = axes.flatten()
    for i, col in enumerate(cols):
        sns.histplot(df[col], kde=True, ax=axes[i], color="#385170")
        axes[i].set_title(f"Distribution of {col}")
    fig.suptitle("Univariate Distributions of Key Numeric Features", fontsize=14, y=1.02)
    fig.tight_layout()
    save_fig(fig, "02_numeric_distributions.png")


# ------------------------------------------------------------------
# 3. Spend / income by segment
# ------------------------------------------------------------------
def plot_spend_by_segment(df):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    sns.boxplot(data=df, x="customer_segment", y="avg_monthly_spend",
                order=SEGMENT_ORDER, palette=SEGMENT_PALETTE, ax=axes[0])
    axes[0].set_title("Monthly Spend by Segment")

    sns.boxplot(data=df, x="customer_segment", y="annual_income",
                order=SEGMENT_ORDER, palette=SEGMENT_PALETTE, ax=axes[1])
    axes[1].set_title("Annual Income by Segment")
    fig.tight_layout()
    save_fig(fig, "03_spend_income_by_segment.png")


# ------------------------------------------------------------------
# 4. Correlation heatmap
# ------------------------------------------------------------------
def plot_correlation_heatmap(df):
    numeric_cols = [
        "age", "annual_income", "months_active", "avg_monthly_spend",
        "purchase_frequency", "avg_order_value", "discount_usage_rate",
        "return_rate", "browsing_time_minutes", "support_interactions",
        "customer_lifetime_value", "spend_to_income_ratio",
        "engagement_score", "loyalty_index"
    ]
    corr = df[numeric_cols].corr()
    fig, ax = plt.subplots(figsize=(11, 9))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0,
                square=True, linewidths=0.5, ax=ax, annot_kws={"size": 7})
    ax.set_title("Correlation Heatmap of Numeric Features")
    fig.tight_layout()
    save_fig(fig, "04_correlation_heatmap.png")


# ------------------------------------------------------------------
# 5. Regional analysis
# ------------------------------------------------------------------
def plot_regional_analysis(df):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    region_segment = pd.crosstab(df["region"], df["customer_segment"], normalize="index")
    region_segment = region_segment[SEGMENT_ORDER]
    region_segment.plot(kind="bar", stacked=True, ax=axes[0],
                         color=[SEGMENT_PALETTE[s] for s in SEGMENT_ORDER])
    axes[0].set_title("Segment Composition by Region (% within region)")
    axes[0].set_ylabel("Proportion")
    axes[0].legend(title="Segment", bbox_to_anchor=(1.02, 1), loc="upper left")

    sns.barplot(data=df, x="region", y="avg_monthly_spend",
                estimator="mean", ax=axes[1], color="#385170")
    axes[1].set_title("Average Monthly Spend by Region")
    fig.tight_layout()
    save_fig(fig, "05_regional_analysis.png")


# ------------------------------------------------------------------
# 6. Engagement vs CLV scatter
# ------------------------------------------------------------------
def plot_engagement_vs_clv(df):
    fig, ax = plt.subplots(figsize=(8, 6))
    sample = df.sample(min(4000, len(df)), random_state=42)
    sns.scatterplot(data=sample, x="engagement_score", y="customer_lifetime_value",
                     hue="customer_segment", hue_order=SEGMENT_ORDER,
                     palette=SEGMENT_PALETTE, alpha=0.6, s=25, ax=ax)
    ax.set_title("Engagement Score vs Customer Lifetime Value")
    ax.legend(title="Segment", bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.tight_layout()
    save_fig(fig, "06_engagement_vs_clv.png")


# ------------------------------------------------------------------
# 7. Payment method preference by segment
# ------------------------------------------------------------------
def plot_payment_by_segment(df):
    ct = pd.crosstab(df["customer_segment"], df["payment_method"], normalize="index")
    ct = ct.reindex(SEGMENT_ORDER)

    fig, ax = plt.subplots(figsize=(8, 6))
    ct.plot(kind="bar", ax=ax, colormap="viridis")
    ax.set_title("Payment Method Preference by Segment")
    ax.set_ylabel("Proportion")
    fig.tight_layout()
    save_fig(fig, "07_payment_by_segment.png")


# ------------------------------------------------------------------
# Insight narrative
# ------------------------------------------------------------------
def generate_insight_narrative(df):
    seg_summary = df.groupby("customer_segment")[
        ["avg_monthly_spend", "annual_income", "purchase_frequency",
         "return_rate", "customer_lifetime_value"]
    ].mean().reindex(SEGMENT_ORDER).round(2)

    high_value_share = (df["customer_segment"] == "High_Value").mean() * 100

    narrative = f"""# EDA Insight Narrative — Retail Customer Intelligence Platform

## 1. Segment Overview
The customer base is dominated by **Occasional** shoppers, followed by Regular,
Loyal, and High_Value customers. High_Value customers make up only
**{high_value_share:.1f}%** of the base but represent the most profitable segment
per the average monthly spend and CLV figures below.

| Segment | Avg Monthly Spend | Avg Annual Income | Avg Purchase Freq | Avg Return Rate | Avg CLV |
|---|---|---|---|---|---|
{chr(10).join(f"| {seg} | {row.avg_monthly_spend} | {row.annual_income} | {row.purchase_frequency} | {row.return_rate} | {row.customer_lifetime_value} |" for seg, row in seg_summary.iterrows())}


"""

    out_path = os.path.join(REPORT_DIR, "eda_insight_narrative.md")
    with open(out_path, "w") as f:
        f.write(narrative)
    print(f"[SAVED] {out_path}")
    return narrative


def run_eda():
    df = load_clean()
    plot_segment_distribution(df)
    plot_numeric_distributions(df)
    plot_spend_by_segment(df)
    plot_correlation_heatmap(df)
    plot_regional_analysis(df)
    plot_engagement_vs_clv(df)
    plot_payment_by_segment(df)
    generate_insight_narrative(df)
    print("\n[DONE] EDA complete. Figures in outputs/figures/, narrative in outputs/reports/")


if __name__ == "__main__":
    run_eda()
