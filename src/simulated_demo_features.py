
import numpy as np
import pandas as pd

RNG = np.random.default_rng(42)

# ------------------------------------------------------------------
# SIMULATED SENTIMENT ANALYSIS
# ------------------------------------------------------------------
_SAMPLE_POSITIVE = [
    "Delivery was fast and the product quality exceeded expectations.",
    "Great customer support, resolved my issue within minutes.",
    "Smooth checkout experience, will definitely shop again.",
    "Loved the packaging and the loyalty rewards program.",
]
_SAMPLE_NEUTRAL = [
    "Product was okay, nothing special but did the job.",
    "Delivery took the expected amount of time.",
    "Average experience overall, matched the description.",
]
_SAMPLE_NEGATIVE = [
    "Return process was frustrating and took too long.",
    "Product did not match the description on the website.",
    "Customer support was slow to respond to my complaint.",
]


def simulate_sentiment_for_customer(customer_row: dict) -> dict:
    """
    Synthetic sentiment generator: skews positive/negative based on the
    customer's real return_rate and support_interactions (higher return
    rate / more support tickets -> more likely to skew negative), so the
    demo at least correlates loosely with real behavioural signals rather
    than being fully random.
    """
    return_rate = customer_row.get("return_rate", 0.1)
    support = customer_row.get("support_interactions", 0)

    negative_weight = min(0.7, return_rate + support * 0.05)
    positive_weight = max(0.1, 0.6 - negative_weight)
    neutral_weight = max(0.05, 1 - negative_weight - positive_weight)
    total = negative_weight + positive_weight + neutral_weight
    probs = [positive_weight / total, neutral_weight / total, negative_weight / total]

    sentiment = RNG.choice(["Positive", "Neutral", "Negative"], p=probs)
    if sentiment == "Positive":
        review = RNG.choice(_SAMPLE_POSITIVE)
        stars = RNG.choice([4, 5])
    elif sentiment == "Neutral":
        review = RNG.choice(_SAMPLE_NEUTRAL)
        stars = 3
    else:
        review = RNG.choice(_SAMPLE_NEGATIVE)
        stars = RNG.choice([1, 2])

    return {"sentiment": sentiment, "stars": int(stars), "sample_review": review}


def simulate_sentiment_distribution(df: pd.DataFrame, n_sample: int = 2000) -> pd.DataFrame:
    sample = df.sample(min(n_sample, len(df)), random_state=42)
    results = sample.apply(lambda r: simulate_sentiment_for_customer(r.to_dict()), axis=1)
    return pd.DataFrame(list(results))


# ------------------------------------------------------------------
# SIMULATED GEOGRAPHIC INTELLIGENCE
# ------------------------------------------------------------------
# The dataset only has 3 region categories (Urban / Semi-Urban / Rural), not
# real city coordinates. We map each region to a small set of illustrative
# Indian city coordinates purely so a map widget has something to plot.
_REGION_CITY_COORDS = {
    "Urban": [("Mumbai", 19.0760, 72.8777), ("Bengaluru", 12.9716, 77.5946),
              ("Delhi", 28.7041, 77.1025), ("Hyderabad", 17.3850, 78.4867)],
    "Semi-Urban": [("Nashik", 19.9975, 73.7898), ("Coimbatore", 11.0168, 76.9558),
                   ("Jaipur", 26.9124, 75.7873), ("Bhopal", 23.2599, 77.4126)],
    "Rural": [("Satara", 17.6805, 73.9932), ("Warangal", 17.9689, 79.5941),
              ("Sitapur", 27.5619, 80.6822), ("Kolhapur", 16.7050, 74.2433)],
}


def simulate_geo_distribution(df: pd.DataFrame) -> pd.DataFrame:
    """Assign each customer a synthetic city + jittered coordinate for map display."""
    rows = []
    for region, group in df.groupby("region"):
        cities = _REGION_CITY_COORDS.get(region, _REGION_CITY_COORDS["Urban"])
        n = len(group)
        assigned = RNG.integers(0, len(cities), size=n)
        for i, (_, row) in enumerate(group.iterrows()):
            city, lat, lon = cities[assigned[i]]
            jitter_lat = lat + RNG.normal(0, 0.15)
            jitter_lon = lon + RNG.normal(0, 0.15)
            rows.append({
                "customer_id": row["customer_id"], "region": region,
                "sim_city": city, "sim_lat": jitter_lat, "sim_lon": jitter_lon,
                "avg_monthly_spend": row["avg_monthly_spend"],
                "customer_segment": row["customer_segment"],
            })
    return pd.DataFrame(rows)


# ------------------------------------------------------------------
# SIMULATED SPENDING FORECAST
# ------------------------------------------------------------------
def simulate_spending_forecast(current_monthly_spend: float, months: int = 6,
                                growth_rate: float = 0.02, volatility: float = 0.05) -> pd.DataFrame:
    """
    Illustrative month-by-month projection using a simple compounding growth
    assumption plus random noise. NOT based on real historical time-series
    data (the dataset has no purchase-date history), so this should be read
    as "what a forecast chart would look like," not a real prediction.
    """
    values = [current_monthly_spend]
    for _ in range(months):
        growth = 1 + growth_rate + RNG.normal(0, volatility)
        values.append(max(0, values[-1] * growth))
    return pd.DataFrame({
        "month": [f"M{i}" for i in range(months + 1)],
        "projected_spend": np.round(values, 2),
    })


# ------------------------------------------------------------------
# SIMULATED REAL-TIME ANALYTICS
# ------------------------------------------------------------------
def simulate_live_snapshot(df: pd.DataFrame) -> dict:
    """
    Illustrative "live" counters derived from real aggregate stats plus small
    random jitter, refreshed on each call — simulates a live dashboard feed
    without an actual event stream backing it.
    """
    base_online = max(5, int(len(df) * 0.001))
    return {
        "customers_online_now": int(base_online + RNG.integers(-3, 4)),
        "products_viewed_last_min": int(RNG.integers(20, 80)),
        "orders_last_min": int(RNG.integers(0, 6)),
        "revenue_today_estimate": round(df["avg_monthly_spend"].mean() * len(df) * 0.001 *
                                         (1 + RNG.normal(0, 0.1)), 2),
    }
