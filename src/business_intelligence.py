
import re
import io
import pandas as pd
from datetime import datetime
from fpdf import FPDF


# ------------------------------------------------------------------
# 1. MARKETING CAMPAIGN RECOMMENDER
# ------------------------------------------------------------------
CAMPAIGN_RULES = {
    "Festival / Seasonal Push": lambda df: df[
        (df["customer_segment"].isin(["Loyal", "High_Value"])) & (df["engagement_score"] > 0.5)
    ],
    "Win-Back At-Risk Customers": lambda df: df[
        df.get("churn_risk_bucket", 0) == 2
    ] if "churn_risk_bucket" in df.columns else df.iloc[0:0],
    "Upsell to Premium": lambda df: df[
        (df["customer_segment"] == "Regular") & (df["annual_income"] > df["annual_income"].median())
    ],
    "Discount-Sensitive Reactivation": lambda df: df[
        (df["discount_usage_rate"] > 0.4) & (df["purchase_frequency"] < df["purchase_frequency"].median())
    ],
    "New Customer Onboarding": lambda df: df[df["months_active"] <= 3],
}


def get_campaign_targets(df: pd.DataFrame, campaign_name: str) -> pd.DataFrame:
    rule = CAMPAIGN_RULES.get(campaign_name)
    if rule is None:
        return df.iloc[0:0]
    return rule(df)


def suggest_channels(df_targets: pd.DataFrame) -> list:
    if df_targets.empty:
        return []
    top_payment = df_targets["payment_method"].mode().iloc[0] if "payment_method" in df_targets else None
    channels = ["Email Campaign"]
    if top_payment == "UPI" or top_payment == "Wallet":
        channels.append("WhatsApp / SMS Campaign")
    if (df_targets["customer_segment"].isin(["Loyal", "High_Value"])).mean() > 0.4:
        channels.append("Dedicated Relationship Manager Outreach")
    else:
        channels.append("Push Notification / App Banner")
    return channels


# ------------------------------------------------------------------
# 2. PERSONALIZED OFFER GENERATOR
# ------------------------------------------------------------------
def generate_offer(customer: dict) -> dict:
    """Rule-based personalized offer for a single customer row (dict)."""
    segment = customer.get("customer_segment", "Occasional")
    discount_usage = customer.get("discount_usage_rate", 0)
    return_rate = customer.get("return_rate", 0)
    engagement = customer.get("engagement_score", 0)
    churn_bucket = customer.get("churn_risk_bucket", None)

    offers = []
    reasons = []

    if segment == "High_Value":
        offers.append("Exclusive Premium Membership Trial")
        reasons.append("Top-tier spender — reward loyalty and protect retention")
    elif segment == "Loyal":
        offers.append("Early Access to New Collections")
        reasons.append("Consistently engaged — reinforce loyalty with exclusivity")
    elif segment == "Regular":
        offers.append("Tiered Cashback (spend more, save more)")
        reasons.append("Room to grow into a higher-value segment")
    else:
        offers.append("Welcome Discount (first 3 purchases)")
        reasons.append("Low engagement so far — incentivize repeat purchase habit")

    if discount_usage > 0.4:
        offers.append("Flash Sale / Coupon Alerts")
        reasons.append("Historically responsive to discounts")

    if churn_bucket == 2:
        offers.append("Win-Back Offer: 20% off next purchase")
        reasons.append("Elevated churn risk — proactive retention incentive")

    if return_rate > 0.35:
        offers.append("Free Size/Fit Consultation or Extended Return Window")
        reasons.append("High return rate — reduce friction rather than push more sales")

    if engagement > 0.7:
        offers.append("Refer-a-Friend Bonus")
        reasons.append("Highly engaged — good candidate for organic advocacy")

    return {"offers": offers, "reasons": reasons}


# ------------------------------------------------------------------
# 3. ALERT SYSTEM
# ------------------------------------------------------------------
def generate_alerts(df: pd.DataFrame, max_alerts: int = 25) -> list:
    alerts = []

    if "churn_risk_bucket" in df.columns:
        high_risk = df[df["churn_risk_bucket"] == 2]
        high_value_at_risk = high_risk[high_risk["customer_segment"].isin(["Loyal", "High_Value"])]
        if len(high_value_at_risk) > 0:
            alerts.append({
                "level": "high",
                "message": f"{len(high_value_at_risk):,} high-value/loyal customers show elevated churn risk"
            })

    if "anomaly_flag" in df.columns:
        n_anom = int(df["anomaly_flag"].sum())
        if n_anom > 0:
            alerts.append({
                "level": "medium",
                "message": f"{n_anom:,} customers flagged with unusual behavioural patterns (review recommended)"
            })

    high_return = df[df["return_rate"] > 0.5]
    if len(high_return) > 0:
        alerts.append({
            "level": "medium",
            "message": f"{len(high_return):,} customers have a return rate above 50%"
        })

    dormant = df[(df["purchase_frequency"] < df["purchase_frequency"].quantile(0.1))]
    if len(dormant) > 0:
        alerts.append({
            "level": "low",
            "message": f"{len(dormant):,} customers fall in the bottom 10% of purchase frequency"
        })

    return alerts[:max_alerts]


# ------------------------------------------------------------------
# 4. DATA ASSISTANT (rule-based NL query engine — not an LLM)
# ------------------------------------------------------------------
def answer_query(df: pd.DataFrame, query: str) -> tuple:
    """
    Returns (answer_text, result_dataframe_or_None).
    Pattern-matches common business questions against the real dataset.
    """
    q = query.lower().strip()

    # "who spent above X" / "spent more than X"
    m = re.search(r"spen[dt].{0,15}(above|more than|over)\s*[₹$]?\s*(\d+)", q)
    if m:
        threshold = float(m.group(2))
        result = df[df["avg_monthly_spend"] > threshold]
        return (f"{len(result):,} customers spend more than ₹{threshold:,.0f} per month on average.", result)

    # "show premium / high value customers"
    if "premium" in q or "high value" in q or "high-value" in q:
        result = df[df["customer_segment"] == "High_Value"]
        return (f"There are {len(result):,} High-Value customers.", result)

    # "which customers are likely to churn" / "churn risk"
    if "churn" in q and "churn_risk_bucket" in df.columns:
        result = df[df["churn_risk_bucket"] == 2]
        return (f"{len(result):,} customers are currently flagged as High churn risk.", result)

    # "loyal customers"
    if "loyal" in q:
        result = df[df["customer_segment"] == "Loyal"]
        return (f"There are {len(result):,} Loyal customers.", result)

    # region-based: "customers in <region>"
    for region in df["region"].unique() if "region" in df.columns else []:
        if region.lower() in q:
            result = df[df["region"] == region]
            return (f"{len(result):,} customers are located in the {region} region.", result)

    # "average spend" / "average income"
    if "average spend" in q or "avg spend" in q:
        val = df["avg_monthly_spend"].mean()
        return (f"Average monthly spend across all customers is ₹{val:,.2f}.", None)
    if "average income" in q or "avg income" in q:
        val = df["annual_income"].mean()
        return (f"Average annual income across all customers is ₹{val:,.2f}.", None)

    # "how many customers"
    if "how many customer" in q or "total customer" in q:
        return (f"There are {len(df):,} customers in the dataset.", None)

    # "anomaly" / "fraud"
    if ("anomaly" in q or "fraud" in q) and "anomaly_flag" in df.columns:
        result = df[df["anomaly_flag"] == 1]
        return (f"{len(result):,} customers are flagged with unusual behavioural patterns.", result)

    return (
        "I couldn't match that to a known query pattern. Try: 'show premium customers', "
        "'who spent above 500', 'which customers are likely to churn', "
        "'customers in Urban region', 'average spend', or 'how many customers'.",
        None
    )


# ------------------------------------------------------------------
# 5. REPORT EXPORTS
# ------------------------------------------------------------------
def _pdf_safe(text) -> str:
    """FPDF's built-in Helvetica font is latin-1 only; strip/replace unsupported chars."""
    text = str(text).replace("₹", "Rs.").replace("–", "-").replace("—", "-")
    return text.encode("latin-1", "ignore").decode("latin-1")


def generate_pdf_report(summary: dict, output_path: str):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 18)
    pdf.cell(0, 12, "Retail Customer Intelligence - Executive Summary", ln=True)
    pdf.set_font("Helvetica", "", 10)
    pdf.cell(0, 8, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}", ln=True)
    pdf.ln(4)

    pdf.set_font("Helvetica", "B", 13)
    pdf.cell(0, 10, "Key Metrics", ln=True)
    pdf.set_font("Helvetica", "", 11)
    for k, v in summary.items():
        if isinstance(v, dict):
            continue
        pdf.set_x(pdf.l_margin)
        pdf.multi_cell(0, 7, _pdf_safe(f"- {k.replace('_', ' ').title()}: {v}"))
    pdf.ln(3)

    if "cluster_labels" in summary:
        pdf.set_font("Helvetica", "B", 13)
        pdf.cell(0, 10, "AI-Discovered Customer Clusters", ln=True)
        pdf.set_font("Helvetica", "", 11)
        for cid, label in summary["cluster_labels"].items():
            pdf.set_x(pdf.l_margin)
            pdf.multi_cell(0, 7, _pdf_safe(f"- Cluster {cid}: {label}"))

    pdf.output(output_path)
    return output_path


def generate_excel_report(df: pd.DataFrame, output_path: str):
    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="Customer Data", index=False)
        seg_summary = df.groupby("customer_segment").agg(
            customers=("customer_id", "count"),
            avg_monthly_spend=("avg_monthly_spend", "mean"),
            avg_income=("annual_income", "mean"),
        ).round(2)
        seg_summary.to_excel(writer, sheet_name="Segment Summary")
    return output_path
