
import os
import json
import numpy as np
import pandas as pd
import streamlit as st
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from business_intelligence import (
    get_campaign_targets, suggest_channels, CAMPAIGN_RULES,
    generate_offer, answer_query,
    generate_pdf_report, generate_excel_report,
)
from advanced_analytics import explain_prediction, BEHAVIOUR_FEATURES, project_future_clv

# ------------------------------------------------------------------
# PATHS
# ------------------------------------------------------------------
BASE = os.path.dirname(__file__)
CLEAN_PATH = os.path.join(BASE, "..", "data", "cleaned_customer_data.csv")
ENRICHED_PATH = os.path.join(BASE, "..", "data", "enriched_customer_data.csv")
RAW_UPLOAD_PATH = os.path.join(BASE, "..", "data", "retail_customer_segmentation.csv")
FIG_DIR = os.path.join(BASE, "..", "outputs", "figures")
MODEL_DIR = os.path.join(BASE, "..", "outputs", "models")
REPORT_DIR = os.path.join(BASE, "..", "outputs", "reports")

SEGMENT_ORDER = ["Occasional", "Regular", "Loyal", "High_Value"]
SEGMENT_COLORS = {
    "Occasional": "#60706F",
    "Regular": "#1F6F54",
    "Loyal": "#C08A2E",
    "High_Value": "#182430",
}
RISK_COLORS = {0: "#1F6F54", 1: "#C08A2E", 2: "#B23A2E"}
RISK_LABELS = {0: "Low", 1: "Medium", 2: "High"}

# ------------------------------------------------------------------
# PAGE CONFIG & STYLE
# ------------------------------------------------------------------
st.set_page_config(
    page_title="Retail Customer Intelligence Platform",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Fraunces:wght@600;700;900&family=Inter:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500;600&display=swap');

    body, .stApp, .stApp .main, .css-1outpf7 {
        background-color: #0F1724 !important;
        color: #E6ECEF !important;
        font-family: 'Inter', sans-serif !important;
    }

    .streamlit-expanderHeader {
        font-family: 'Fraunces', serif !important;
        letter-spacing: 0.03em;
        color: #E6ECEF !important;
    }

    .stApp .css-1d391kg {
        background-color: #0F1724 !important;
    }

    section[data-testid='stSidebar'] {
        background-color: #111A2A !important;
        color: #E6ECEF !important;
        border-right: 1px solid rgba(230, 236, 239, 0.10) !important;
    }

    .stSidebar .css-1d391kg, .stSidebar .css-1siy2j7 {
        color: #E6ECEF !important;
    }

    .stSidebar h2, .stSidebar h3, .stSidebar p, .stSidebar label {
        color: #E6ECEF !important;
    }

    .stMetric {
        background-color: #151F2F !important;
        color: #E6ECEF !important;
        padding: 18px !important;
        border-radius: 20px !important;
        border: 1px solid rgba(230, 236, 239, 0.10) !important;
        box-shadow: 0 18px 36px rgba(0, 0, 0, 0.45);
        font-family: 'IBM Plex Mono', monospace !important;
    }

    .stMetric * {
        color: #E6ECEF !important;
        font-family: 'IBM Plex Mono', monospace !important;
    }

    .segment-badge, .risk-badge {
        display: inline-block;
        padding: 6px 18px;
        border-radius: 999px;
        color: #E6ECEF;
        font-weight: 700;
        font-size: 0.95rem;
        font-family: 'Fraunces', serif;
        text-transform: uppercase;
        letter-spacing: 0.04em;
    }

    .stButton>button, .stDownloadButton>button,
    button[kind='primary'] {
        background-color: #1F6F54 !important;
        color: #F3F4F1 !important;
        border: 1px solid #C08A2E !important;
        border-radius: 14px !important;
        padding: 12px 20px !important;
        font-family: 'Inter', sans-serif !important;
        font-weight: 700 !important;
        letter-spacing: 0.02em;
        box-shadow: 0 14px 28px rgba(31, 111, 84, 0.32);
        transition: transform 0.18s ease, box-shadow 0.18s ease, background-color 0.18s ease;
    }

    .stButton>button:hover, .stDownloadButton>button:hover {
        background-color: #18563f !important;
        border-color: #C08A2E !important;
        transform: translateY(-1px);
        box-shadow: 0 20px 38px rgba(31, 111, 84, 0.44);
    }

    .css-1wulxc2, .css-1kyxreq { /* table container improvements */
        border-radius: 18px !important;
        background-color: #151F2F !important;
        border: 1px solid rgba(230, 236, 239, 0.08) !important;
        box-shadow: 0 18px 38px rgba(0, 0, 0, 0.30);
    }

    .css-10trblm, .css-18e3th9, .css-1v3fvcr {
        font-family: 'Inter', sans-serif !important;
        color: #E6ECEF !important;
    }

    .stTextInput input, .stNumberInput input, .stSelectbox select,
    .stTextArea textarea {
        background-color: #101A24 !important;
        color: #E6ECEF !important;
        border: 1px solid rgba(230, 236, 239, 0.12) !important;
        border-radius: 12px !important;
    }

    h1, h2, h3, strong {
        color: #E6ECEF !important;
        font-family: 'Fraunces', serif !important;
    }

    .app-view-container .main {
        padding-top: 18px !important;
    }
</style>
""", unsafe_allow_html=True)


# ------------------------------------------------------------------
# CACHED LOADERS
# ------------------------------------------------------------------
@st.cache_data
def load_clean_data():
    if not os.path.exists(CLEAN_PATH):
        return None
    return pd.read_csv(CLEAN_PATH)


@st.cache_data
def load_enriched_data():
    if os.path.exists(ENRICHED_PATH):
        return pd.read_csv(ENRICHED_PATH)
    return None


@st.cache_resource
def load_segment_model():
    paths = [os.path.join(MODEL_DIR, f) for f in
             ["best_model.joblib", "scaler.joblib", "feature_columns.json", "segment_mapping.json"]]
    if not all(os.path.exists(p) for p in paths):
        return None, None, None, None
    model = joblib.load(paths[0])
    scaler = joblib.load(paths[1])
    feature_cols = json.load(open(paths[2]))
    segment_mapping = json.load(open(paths[3]))
    return model, scaler, feature_cols, segment_mapping


@st.cache_resource
def load_churn_model():
    paths = [os.path.join(MODEL_DIR, f) for f in ["churn_model.joblib", "churn_scaler.joblib", "churn_features.json"]]
    if not all(os.path.exists(p) for p in paths):
        return None, None, None
    return joblib.load(paths[0]), joblib.load(paths[1]), json.load(open(paths[2]))


@st.cache_resource
def load_clv_model():
    paths = [os.path.join(MODEL_DIR, f) for f in ["clv_forecast_model.joblib", "clv_drivers.json"]]
    if not all(os.path.exists(p) for p in paths):
        return None, None
    return joblib.load(paths[0]), json.load(open(paths[1]))


@st.cache_data
def load_advanced_summary():
    p = os.path.join(REPORT_DIR, "advanced_analytics_summary.json")
    if os.path.exists(p):
        return json.load(open(p))
    return None


@st.cache_data
def load_cluster_profiles():
    p = os.path.join(REPORT_DIR, "cluster_profiles.csv")
    if os.path.exists(p):
        return pd.read_csv(p, index_col=0)
    return None


@st.cache_data
def feature_reference_stats(_df, features):
    ref = {}
    for f in features:
        if f in _df.columns:
            ref[f] = _df[f].mean()
            ref[f"{f}_std"] = _df[f].std() or 1
    return ref


df_clean = load_clean_data()
df = load_enriched_data()
if df is None:
    df = df_clean  # fall back gracefully if advanced_analytics.py hasn't run yet

model, scaler, feature_cols, segment_mapping = load_segment_model()
churn_model, churn_scaler, churn_features = load_churn_model()
clv_model, clv_drivers = load_clv_model()
adv_summary = load_advanced_summary()
cluster_profiles = load_cluster_profiles()

# ------------------------------------------------------------------
# ROLE-BASED VIEW (lightweight demo — not real authentication)
# ------------------------------------------------------------------
st.sidebar.title("Retail Customer Intelligence Platform")
st.sidebar.caption("AI-Powered customer analytics and recommendations")

role = st.sidebar.selectbox(
    "Logged in as", ["Analyst", "Marketing Manager", "Sales Executive", "Admin"],
    help="Demo role-based view — restricts the Admin Panel only. Not a secured login system."
)

PAGES = [
    "Dashboard", "Customer Explorer", "Segmentation",
    "EDA Insights", "Churn Risk & Explainable AI", "CLV Forecast",
    "Predict Segment", "Marketing & Offers",
    "Data Assistant", "Anomaly Detection", "Model Performance",
    "Reports & Export", "Admin Panel",
]
page = st.sidebar.radio("Navigate", PAGES)

if df is None:
    st.error("Cleaned dataset not found. Please run `src/data_cleaning.py` first.")
    st.stop()

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


# ================================================================================
# PAGE: DASHBOARD
# ================================================================================
if page == "Dashboard":
    st.title("Executive Dashboard")
    st.caption("KPI overview computed from the actual customer dataset")

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total Customers", f"{len(df):,}")
    c2.metric("Total Revenue (est.)", f"₹{(df['avg_monthly_spend'].sum()):,.0f}")
    c3.metric("Avg Monthly Spend", f"₹{df['avg_monthly_spend'].mean():,.0f}")
    c4.metric("High-Value Share", f"{(df['customer_segment']=='High_Value').mean()*100:.1f}%")
    if "churn_risk_bucket" in df.columns:
        churn_pct = (df["churn_risk_bucket"] == 2).mean() * 100
        c5.metric("High Churn Risk", f"{churn_pct:.1f}%")
    else:
        c5.metric("High Churn Risk", "run advanced_analytics.py")

    st.markdown("### Segment Breakdown")
    seg_counts = df["customer_segment"].value_counts().reindex(SEGMENT_ORDER)
    cols = st.columns(4)
    for i, seg in enumerate(SEGMENT_ORDER):
        with cols[i]:
            st.markdown(f'<span class="segment-badge" style="background-color:{SEGMENT_COLORS[seg]};">{seg}</span>',
                        unsafe_allow_html=True)
            st.metric("", f"{seg_counts[seg]:,}", label_visibility="collapsed")

    if adv_summary:
        st.markdown("---")
        st.markdown("### AI Model Health")
        m1, m2, m3 = st.columns(3)
        m1.metric("Churn Model Accuracy", f"{adv_summary['churn_model_metrics']['accuracy']*100:.1f}%")
        m2.metric("CLV Forecast R²", f"{adv_summary['clv_forecast_metrics']['r2']:.2f}")
        m3.metric("Anomalies Flagged", f"{adv_summary['n_anomalies_flagged']:,}")


# ================================================================================
# PAGE: CUSTOMER EXPLORER
# ================================================================================
elif page == "Customer Explorer":
    st.title("Customer Explorer")
    st.caption("Search any customer to see their full profile, summary, and recommended actions")

    cust_id = st.selectbox("Search by Customer ID", options=sorted(df["customer_id"].unique()), index=0)
    row = df[df["customer_id"] == cust_id].iloc[0]

    seg = row["customer_segment"]
    st.markdown(f'<span class="segment-badge" style="background-color:{SEGMENT_COLORS[seg]}; font-size:1.1rem;">{seg}</span>',
                unsafe_allow_html=True)
    if "churn_risk_bucket" in row:
        risk = int(row["churn_risk_bucket"])
        st.markdown(f'&nbsp;&nbsp;<span class="risk-badge" style="background-color:{RISK_COLORS[risk]};">Churn Risk: {RISK_LABELS[risk]}</span>',
                    unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("**Profile**")
        st.write(f"Age: {row['age']}")
        st.write(f"Annual Income: ₹{row['annual_income']:,.0f}")
        st.write(f"Region: {row['region']}")
        st.write(f"Payment Method: {row['payment_method']}")
        st.write(f"Months Active: {row['months_active']}")
    with c2:
        st.markdown("**Behaviour**")
        st.write(f"Avg Monthly Spend: ₹{row['avg_monthly_spend']:,.2f}")
        st.write(f"Purchase Frequency: {row['purchase_frequency']:.2f}/mo")
        st.write(f"Avg Order Value: ₹{row['avg_order_value']:,.2f}")
        st.write(f"Return Rate: {row['return_rate']*100:.1f}%")
        st.write(f"Discount Usage: {row['discount_usage_rate']*100:.1f}%")
    with c3:
        st.markdown("**Derived Metrics**")
        st.write(f"Lifetime Value (to date): ₹{row['customer_lifetime_value']:,.0f}")
        st.write(f"Engagement Score: {row['engagement_score']:.2f}")
        if "ai_cluster_label" in row:
            st.write(f"AI Cluster: {row['ai_cluster_label']}")
        if "anomaly_flag" in row:
            st.write(f"Anomaly Flag: {'Yes' if row['anomaly_flag']==1 else 'No'}")

    st.markdown("---")
    st.markdown("### AI Summary")
    avg_spend = df["avg_monthly_spend"].mean()
    spend_diff_pct = (row["avg_monthly_spend"] - avg_spend) / avg_spend * 100
    summary_bits = []
    summary_bits.append(
        f"This customer spends {abs(spend_diff_pct):.0f}% "
        f"{'more' if spend_diff_pct > 0 else 'less'} than the average customer."
    )
    if row["engagement_score"] > 0.6:
        summary_bits.append("They show high platform engagement (frequent browsing and purchasing).")
    elif row["engagement_score"] < 0.3:
        summary_bits.append("Their engagement is currently low, suggesting a need for re-activation.")
    if "churn_risk_bucket" in row and int(row["churn_risk_bucket"]) == 2:
        summary_bits.append("They are currently flagged as **high churn risk** — proactive retention is recommended.")
    if row["return_rate"] > 0.35:
        summary_bits.append("Their return rate is notably high, which may be worth investigating for fit/quality issues.")
    st.info(" ".join(summary_bits))

    st.markdown("### Recommended Offer")
    offer = generate_offer(row.to_dict())
    for o, r in zip(offer["offers"], offer["reasons"]):
        st.markdown(f"- **{o}** — _{r}_")

    if "churn_risk_bucket" in row and clv_model is not None:
        st.markdown("### CLV Forecast")
        drivers_vals = np.array([[row.get(d, 0) for d in clv_drivers]], dtype=float)
        pred_monthly = clv_model.predict(drivers_vals)[0]
        forecast_clv = project_future_clv(pred_monthly, row["months_active"], row["engagement_score"])
        st.metric("Projected next-12-month value", f"₹{forecast_clv:,.0f}",
                   help="Estimation model — see CLV Forecast page for methodology")


# ================================================================================
# PAGE: SEGMENTATION
# ================================================================================
elif page == "Segmentation":
    st.title("Customer Segmentation")
    tab1, tab2 = st.tabs(["Business Segments (labeled)", "AI-Discovered Clusters (unsupervised)"])

    with tab1:
        st.markdown("These are the segment labels provided in the source dataset.")
        seg_summary = df.groupby("customer_segment")[
            ["avg_monthly_spend", "annual_income", "purchase_frequency", "customer_lifetime_value"]
        ].mean().round(2).reindex(SEGMENT_ORDER)
        st.dataframe(seg_summary, use_container_width=True)
        fpath = os.path.join(FIG_DIR, "01_segment_distribution.png")
        if os.path.exists(fpath):
            st.image(fpath, use_container_width=True)

    with tab2:
        st.markdown("These clusters were discovered independently using **KMeans** on behavioural "
                     "features (spend, frequency, engagement, income, etc.) — without using the "
                     "provided segment label at all. Useful for cross-checking whether the business "
                     "segments align with natural behavioural groupings.")
        if cluster_profiles is not None:
            display_cols = [c for c in cluster_profiles.columns if c != "label"]
            st.dataframe(cluster_profiles[["label"] + display_cols], use_container_width=True)

            if "ai_cluster_label" in df.columns:
                fig, ax = plt.subplots(figsize=(8, 5))
                counts = df["ai_cluster_label"].value_counts()
                sns.barplot(x=counts.values, y=counts.index, color="#385170", ax=ax)
                ax.set_title("AI Cluster Sizes")
                ax.set_xlabel("Number of Customers")
                st.pyplot(fig)

                st.markdown("#### Cross-tab: Business Segment vs AI Cluster")
                ct = pd.crosstab(df["customer_segment"], df["ai_cluster_label"])
                st.dataframe(ct, use_container_width=True)
        else:
            st.warning("Cluster data not found. Run `src/advanced_analytics.py` first.")


# ================================================================================
# PAGE: EDA INSIGHTS (original)
# ================================================================================
elif page == "EDA Insights":
    st.title("Exploratory Data Analysis & Insights")
    st.caption("Visualizations and narrative produced by the EDA & Insight team")

    figure_files = [
        ("01_segment_distribution.png", "Customer Segment Distribution"),
        ("02_numeric_distributions.png", "Numeric Feature Distributions"),
        ("03_spend_income_by_segment.png", "Spend & Income by Segment"),
        ("04_correlation_heatmap.png", "Correlation Heatmap"),
        ("05_regional_analysis.png", "Regional Analysis"),
        ("06_engagement_vs_clv.png", "Engagement vs. Customer Lifetime Value"),
        ("07_payment_by_segment.png", "Payment Method Preference by Segment"),
    ]
    tabs = st.tabs([t for _, t in figure_files])
    for tab, (fname, title) in zip(tabs, figure_files):
        with tab:
            fpath = os.path.join(FIG_DIR, fname)
            if os.path.exists(fpath):
                st.image(fpath, use_container_width=True, caption=title)
            else:
                st.warning(f"Figure not found: {fname}. Run src/eda_visualization.py first.")

    narrative_path = os.path.join(REPORT_DIR, "eda_insight_narrative.md")
    if os.path.exists(narrative_path):
        st.markdown("---")
        st.markdown("### 📝 Insight Narrative")
        st.markdown(open(narrative_path).read())


# ================================================================================
# PAGE: CHURN RISK & EXPLAINABLE AI
# ================================================================================
elif page == "Churn Risk & Explainable AI":
    st.title("Churn Risk & Explainable AI")

    if churn_model is None or "churn_risk_bucket" not in df.columns:
        st.error("Churn model not found. Run `src/advanced_analytics.py` first.")
    else:
        with st.expander("ℹ️ How churn risk is defined (methodology)", expanded=False):
            st.markdown("""
            This dataset has **no explicit churn event or date**, so churn risk is built from a
            transparent proxy score combining purchase frequency, engagement score, tenure, and
            return rate (documented in `advanced_analytics.py`). Customers are bucketed into
            Low / Medium / High risk tertiles, and a classifier is trained on that label so it can
            be applied to explain *which behavioural features* drive risk.
            """)

        risk_counts = df["churn_risk_bucket"].map(RISK_LABELS).value_counts().reindex(["Low", "Medium", "High"])
        c1, c2, c3 = st.columns(3)
        c1.metric("Low Risk", f"{risk_counts['Low']:,}")
        c2.metric("Medium Risk", f"{risk_counts['Medium']:,}")
        c3.metric("High Risk", f"{risk_counts['High']:,}")

        fig, ax = plt.subplots(figsize=(7, 4))
        sns.barplot(x=risk_counts.index, y=risk_counts.values,
                    palette=[RISK_COLORS[0], RISK_COLORS[1], RISK_COLORS[2]], ax=ax)
        ax.set_title("Churn Risk Distribution")
        st.pyplot(fig)

        st.markdown("### Global Feature Importance (what drives churn risk)")
        importances = churn_model.feature_importances_
        imp_df = pd.DataFrame({"feature": churn_features, "importance": importances}).sort_values(
            "importance", ascending=False)
        fig2, ax2 = plt.subplots(figsize=(8, 5))
        sns.barplot(data=imp_df, x="importance", y="feature", color="#e05c5c", ax=ax2)
        ax2.set_title("Churn Risk — Feature Importance")
        st.pyplot(fig2)

        st.markdown("### 🔍 Explain a Specific Customer")
        cust_id = st.selectbox("Customer ID", options=sorted(df["customer_id"].unique()), key="explain_cust")
        row = df[df["customer_id"] == cust_id].iloc[0]
        ref = feature_reference_stats(df, churn_features)
        exp_df = explain_prediction(row.to_dict(), churn_features, importances, ref)
        risk = RISK_LABELS[int(row["churn_risk_bucket"])]
        st.write(f"**Predicted churn risk:** {risk}")
        st.dataframe(exp_df.head(8), use_container_width=True)
        st.caption("Positive contribution = pushes risk higher; negative = pushes risk lower. "
                   "This is a transparent deviation-from-average explanation, not exact SHAP values.")


# ================================================================================
# PAGE: CLV FORECAST
# ================================================================================
elif page == "CLV Forecast":
    st.title("Customer Lifetime Value Forecast")

    if clv_model is None:
        st.error("CLV model not found. Run `src/advanced_analytics.py` first.")
    else:
        with st.expander("ℹ️ Methodology", expanded=False):
            st.markdown("""
            A regression model predicts *next-period monthly spend* from non-circular behavioural
            drivers (income, order value, discount usage, return rate, browsing time, engagement).
            Forecast CLV = predicted monthly spend × an engagement-weighted expected additional
            tenure (higher engagement → longer expected retention). This is an **estimation model**,
            not a guarantee.
            """)

        cust_id = st.selectbox("Customer ID", options=sorted(df["customer_id"].unique()), key="clv_cust")
        row = df[df["customer_id"] == cust_id].iloc[0]
        drivers_vals = np.array([[row.get(d, 0) for d in clv_drivers]], dtype=float)
        pred_monthly = clv_model.predict(drivers_vals)[0]
        forecast_clv = project_future_clv(pred_monthly, row["months_active"], row["engagement_score"])

        c1, c2, c3 = st.columns(3)
        c1.metric("Historical CLV (to date)", f"₹{row['customer_lifetime_value']:,.0f}")
        c2.metric("Predicted Next-Month Spend", f"₹{pred_monthly:,.2f}")
        c3.metric("Projected 12-Month Forward CLV", f"₹{forecast_clv:,.0f}")

        st.markdown("### CLV Distribution Across All Customers")
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.histplot(df["customer_lifetime_value"], kde=True, color="#385170", ax=ax)
        ax.axvline(row["customer_lifetime_value"], color="#e05c5c", linestyle="--", label="Selected customer")
        ax.legend()
        ax.set_title("Historical CLV Distribution")
        st.pyplot(fig)

        st.markdown("### Top 10 Customers by Historical CLV")
        top10 = df.nlargest(10, "customer_lifetime_value")[
            ["customer_id", "customer_segment", "customer_lifetime_value", "avg_monthly_spend"]]
        st.dataframe(top10, use_container_width=True)


# ================================================================================
# PAGE: PREDICT SEGMENT (original)
# ================================================================================
elif page == "Predict Segment":
    st.title("Predict a Customer's Segment")
    st.caption("Live inference powered by the trained customer segment model")

    if model is None:
        st.error("Trained model not found. Please run `src/modeling.py` first.")
        st.stop()

    with st.form("predict_form"):
        c1, c2, c3 = st.columns(3)
        with c1:
            age = st.number_input("Age", 18, 70, 35)
            annual_income = st.number_input("Annual Income (₹)", 0, 500000, 60000, step=1000)
            months_active = st.number_input("Months Active", 0, 120, 24)
            purchase_frequency = st.number_input("Purchase Frequency (per month)", 0.0, 20.0, 2.5)
        with c2:
            avg_monthly_spend = st.number_input("Avg Monthly Spend (₹)", 0.0, 5000.0, 300.0)
            avg_order_value = st.number_input("Avg Order Value (₹)", 0.0, 2000.0, 100.0)
            discount_usage_rate = st.slider("Discount Usage Rate", 0.0, 1.0, 0.2)
            return_rate = st.slider("Return Rate", 0.0, 1.0, 0.1)
        with c3:
            browsing_time_minutes = st.number_input("Browsing Time (min/session)", 0.0, 400.0, 60.0)
            support_interactions = st.number_input("Support Interactions", 0, 15, 1)
            payment_method = st.selectbox("Payment Method", ["Card", "UPI", "Wallet"])
            region = st.selectbox("Region", ["Urban", "Semi-Urban", "Rural"])
        submitted = st.form_submit_button("Predict Segment", use_container_width=True)

    if submitted:
        clv = avg_monthly_spend * months_active
        spend_to_income = (avg_monthly_spend * 12) / annual_income if annual_income > 0 else 0
        bt_min, bt_max = df["browsing_time_minutes"].min(), df["browsing_time_minutes"].max()
        pf_min, pf_max = df["purchase_frequency"].min(), df["purchase_frequency"].max()
        bt_norm = (browsing_time_minutes - bt_min) / (bt_max - bt_min) if bt_max > bt_min else 0
        pf_norm = (purchase_frequency - pf_min) / (pf_max - pf_min) if pf_max > pf_min else 0
        engagement_score = (bt_norm + pf_norm) / 2
        loyalty_index = months_active * purchase_frequency
        is_high_returner = int(return_rate >= df["return_rate"].quantile(0.75))
        avg_spend_per_order = avg_monthly_spend / purchase_frequency if purchase_frequency > 0 else 0

        row = {
            "age": age, "annual_income": annual_income, "months_active": months_active,
            "avg_monthly_spend": avg_monthly_spend, "purchase_frequency": purchase_frequency,
            "avg_order_value": avg_order_value, "discount_usage_rate": discount_usage_rate,
            "return_rate": return_rate, "browsing_time_minutes": browsing_time_minutes,
            "support_interactions": support_interactions,
            "customer_lifetime_value": clv, "spend_to_income_ratio": spend_to_income,
            "engagement_score": engagement_score, "loyalty_index": loyalty_index,
            "is_high_returner": is_high_returner, "avg_spend_per_order": avg_spend_per_order,
            "pay_Card": 1 if payment_method == "Card" else 0,
            "pay_UPI": 1 if payment_method == "UPI" else 0,
            "pay_Wallet": 1 if payment_method == "Wallet" else 0,
            "region_Rural": 1 if region == "Rural" else 0,
            "region_Semi-Urban": 1 if region == "Semi-Urban" else 0,
            "region_Urban": 1 if region == "Urban" else 0,
        }
        X_input = np.array([[row.get(col, 0) for col in feature_cols]], dtype=float)
        X_scaled = scaler.transform(X_input)
        pred_class = int(model.predict(X_scaled)[0])
        pred_segment = segment_mapping[str(pred_class)]
        proba = model.predict_proba(X_scaled)[0] if hasattr(model, "predict_proba") else None

        st.markdown("### Result")
        st.markdown(f'<span class="segment-badge" style="background-color:{SEGMENT_COLORS[pred_segment]}; '
                     f'font-size:1.3rem; padding:8px 24px;">Predicted Segment: {pred_segment}</span>',
                     unsafe_allow_html=True)

        if proba is not None:
            st.markdown("#### Prediction Confidence by Segment")
            proba_df = pd.DataFrame({"Segment": [segment_mapping[str(i)] for i in range(len(proba))],
                                      "Probability": proba})
            fig, ax = plt.subplots(figsize=(7, 3.5))
            sns.barplot(data=proba_df, x="Segment", y="Probability", order=SEGMENT_ORDER,
                        palette=[SEGMENT_COLORS[s] for s in SEGMENT_ORDER], ax=ax)
            ax.set_ylim(0, 1)
            for i, v in enumerate(proba_df.set_index("Segment").reindex(SEGMENT_ORDER)["Probability"]):
                ax.text(i, v + 0.02, f"{v:.1%}", ha="center", fontweight="bold")
            st.pyplot(fig)


# ================================================================================
# PAGE: MARKETING & OFFERS
# ================================================================================
elif page == "Marketing & Offers":
    st.title("Marketing Campaign Recommender & Offer Generator")

    tab1, tab2 = st.tabs(["Campaign Targeting", "Personalized Offer Generator"])

    with tab1:
        campaign = st.selectbox("Choose a campaign type", list(CAMPAIGN_RULES.keys()))
        targets = get_campaign_targets(df, campaign)
        st.metric("Target Audience Size", f"{len(targets):,}")

        if len(targets) > 0:
            channels = suggest_channels(targets)
            st.markdown("**Suggested channels:** " + ", ".join(channels))
            st.dataframe(
                targets[["customer_id", "customer_segment", "region", "avg_monthly_spend", "payment_method"]].head(200),
                use_container_width=True
            )
            csv = targets.to_csv(index=False).encode("utf-8")
            st.download_button("Download full target list (CSV)", csv, f"{campaign.replace(' ', '_')}_targets.csv")
        else:
            st.info("No customers currently match this campaign's criteria (or run advanced_analytics.py for churn-based campaigns).")

    with tab2:
        cust_id = st.selectbox("Customer ID", options=sorted(df["customer_id"].unique()), key="offer_cust")
        row = df[df["customer_id"] == cust_id].iloc[0]
        offer = generate_offer(row.to_dict())
        st.markdown(f"**Segment:** {row['customer_segment']}")
        for o, r in zip(offer["offers"], offer["reasons"]):
            st.markdown(f"- **{o}** — _{r}_")


# ================================================================================
# PAGE: DATA ASSISTANT (rule-based chatbot)
# ================================================================================
elif page == "Data Assistant":
    st.title("Data Assistant")
    st.caption("Ask questions about your customers in plain English. This is a rule-based query engine over your real data.")

    with st.expander("Example questions"):
        st.markdown("""
        - Who spent above 500?
        - Show premium customers
        - Which customers are likely to churn?
        - Customers in Urban region
        - What is the average spend?
        - How many customers are there?
        """)

    query = st.text_input("Ask a question about your customers")
    if st.button("Ask") and query:
        answer, result = answer_query(df, query)
        st.session_state.chat_history.append((query, answer, result))

    for q, a, result in reversed(st.session_state.chat_history):
        st.markdown(f"**You:** {q}")
        st.markdown(f"**Assistant:** {a}")
        if result is not None and len(result) > 0:
            st.dataframe(result.head(100), use_container_width=True)
        st.markdown("---")


# ================================================================================
# PAGE: ANOMALY DETECTION
# ================================================================================
elif page == "Anomaly Detection":
    st.title("Anomaly Detection (Behavioural Outliers)")
    if "anomaly_flag" not in df.columns:
        st.error("Anomaly model not found. Run `src/advanced_analytics.py` first.")
    else:
        st.info("There is no ground-truth fraud label in this dataset. IsolationForest flags "
                "customers whose behavioural pattern is statistically unusual — treat this as a "
                "**prioritization signal for manual review**, not a confirmed verdict.")

        n_flagged = int(df["anomaly_flag"].sum())
        st.metric("Customers Flagged", f"{n_flagged:,} ({n_flagged/len(df)*100:.1f}%)")

        flagged = df[df["anomaly_flag"] == 1].sort_values("anomaly_score", ascending=False)
        st.dataframe(
            flagged[["customer_id", "customer_segment", "avg_monthly_spend", "return_rate",
                      "discount_usage_rate", "support_interactions", "anomaly_score"]].head(100),
            use_container_width=True
        )
        csv = flagged.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Download flagged customers (CSV)", csv, "anomaly_flagged_customers.csv")


# ================================================================================
# PAGE: MODEL PERFORMANCE (original)
# ================================================================================
elif page == "Model Performance":
    st.title("Model Performance")
    st.caption("Saved evaluation results from the trained models")

    results_path = os.path.join(REPORT_DIR, "model_comparison_results.csv")
    if os.path.exists(results_path):
        st.dataframe(pd.read_csv(results_path), use_container_width=True)
    else:
        st.warning("Model comparison results not found. Run src/modeling.py first.")

    perf_figs = [("09_model_comparison.png", "Model Comparison — Metrics"),
                 ("08_confusion_matrix.png", "Confusion Matrix (Best Model)"),
                 ("10_feature_importance.png", "Feature Importance (Best Model)")]
    cols = st.columns(3)
    for col, (fname, title) in zip(cols, perf_figs):
        fpath = os.path.join(FIG_DIR, fname)
        with col:
            if os.path.exists(fpath):
                st.image(fpath, use_container_width=True, caption=title)
            else:
                st.info(f"{fname} not generated yet.")

    if adv_summary:
        st.markdown("---")
        st.markdown("### Extended Model Metrics")
        c1, c2 = st.columns(2)
        c1.json(adv_summary["churn_model_metrics"])
        c2.json(adv_summary["clv_forecast_metrics"])


# ================================================================================
# PAGE: REPORTS & EXPORT
# ================================================================================
elif page == "Reports & Export":
    st.title("Reports & Export Center")

    st.markdown("### Generate Executive Report")
    if adv_summary and st.button("Generate PDF Executive Summary"):
        out_path = os.path.join(REPORT_DIR, "executive_summary.pdf")
        generate_pdf_report(adv_summary, out_path)
        with open(out_path, "rb") as f:
            st.download_button("⬇️ Download PDF", f, file_name="executive_summary.pdf")

    if st.button("Generate Excel Workbook (full dataset + segment summary)"):
        out_path = os.path.join(REPORT_DIR, "customer_workbook.xlsx")
        generate_excel_report(df, out_path)
        with open(out_path, "rb") as f:
            st.download_button("⬇️ Download Excel", f, file_name="customer_workbook.xlsx")

    st.markdown("---")
    st.markdown("### Written Reports")
    report_files = {
        "EDA Insight Narrative": "eda_insight_narrative.md",
        "Modelling Report": "modeling_report.md",
        "Model Comparison Results (CSV)": "model_comparison_results.csv",
        "Cluster Profiles (CSV)": "cluster_profiles.csv",
    }
    for label, fname in report_files.items():
        fpath = os.path.join(REPORT_DIR, fname)
        if os.path.exists(fpath):
            with open(fpath, "rb") as f:
                st.download_button(f"Download: {label}", f, file_name=fname, key=f"dl_{fname}")
        else:
            st.info(f"{label} not yet generated ({fname}).")


# ================================================================================
# PAGE: ADMIN PANEL
# ================================================================================
elif page == "Admin Panel":
    st.title("⚙️ Admin Panel")

    if role != "Admin":
        st.warning("🔒 Access restricted. Switch 'Logged in as' to **Admin** in the sidebar to use this panel.")
        st.stop()

    st.markdown("### 📤 Upload a New Dataset")
    uploaded = st.file_uploader("Upload a replacement CSV (same schema as retail_customer_segmentation.csv)",
                                  type="csv")
    if uploaded is not None:
        if st.button("Save uploaded file as the active dataset"):
            with open(RAW_UPLOAD_PATH, "wb") as f:
                f.write(uploaded.getbuffer())
            st.success(f"Saved to {RAW_UPLOAD_PATH}. Retrain the pipeline below to apply it.")

    st.markdown("---")
    st.markdown("### 🔁 Retrain Pipeline")
    st.caption("Runs the full pipeline in order: cleaning → EDA → modelling → advanced analytics. "
               "Takes under a minute on this dataset size.")
    if st.button("▶️ Retrain everything now"):
        with st.spinner("Running data_cleaning.py ..."):
            import data_cleaning
            data_cleaning.run_pipeline()
        with st.spinner("Running eda_visualization.py ..."):
            import eda_visualization
            eda_visualization.run_eda()
        with st.spinner("Running modeling.py ..."):
            import modeling
            modeling.run_modeling()
        with st.spinner("Running advanced_analytics.py ..."):
            import advanced_analytics
            advanced_analytics.run_advanced_analytics()
        st.cache_data.clear()
        st.cache_resource.clear()
        st.success("✅ Retraining complete. Reload the page to see fresh results everywhere.")

    st.markdown("---")
    st.markdown("### User Management")
    st.caption("Illustrative only — this app has no real authentication backend.")
    demo_users = pd.DataFrame({
        "user": ["priya.analyst", "raj.marketing", "asha.sales", "admin"],
        "role": ["Analyst", "Marketing Manager", "Sales Executive", "Admin"],
        "status": ["Active", "Active", "Active", "Active"],
    })
    st.dataframe(demo_users, use_container_width=True)

    st.markdown("---")
    st.markdown("### ⚙️ Thresholds")
    st.caption("Adjust illustrative business thresholds (not yet wired into retraining).")
    st.slider("High churn-risk cutoff (top %)", 10, 50, 33)
    st.slider("Anomaly contamination rate (%)", 1, 10, 3)
