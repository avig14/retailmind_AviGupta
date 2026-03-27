"""
core/analytics.py — Pure Python analytics layer for RetailMind.
Zero Streamlit imports. Zero agent imports.
All business formulas and data-processing functions live here.
"""

import pandas as pd
from typing import Optional

# ── Business Constants ────────────────────────────────────────────────────────
CRITICAL_DAYS: int = 7          # Days to stockout → Critical status
LOW_DAYS: int = 14              # Days to stockout → Low status
LOW_MARGIN_FLAG: float = 20.0   # % — flag product if gross margin < 20%
BRIEFING_MARGIN_FLAG: float = 25.0  # % — daily briefing flags if < 25%
PREMIUM_BAND: float = 1.20      # price > category_avg * 1.20 → Premium
BUDGET_BAND: float = 0.80       # price < category_avg * 0.80 → Budget
REVENUE_THRESHOLD_DAYS: int = 7  # default days for restock revenue-at-risk calc


# ── CSV Loaders ───────────────────────────────────────────────────────────────

def load_products(path: str = "data/retailmind_products.csv") -> pd.DataFrame:
    """Load and return the products DataFrame with correct dtypes."""
    df = pd.read_csv(path)
    df["launch_date"] = pd.to_datetime(df["launch_date"])
    df["price"] = df["price"].astype(float)
    df["cost"] = df["cost"].astype(float)
    df["avg_daily_sales"] = df["avg_daily_sales"].astype(float)
    df["avg_rating"] = df["avg_rating"].astype(float)
    return df


def load_reviews(path: str = "data/retailmind_reviews.csv") -> pd.DataFrame:
    """Load and return the reviews DataFrame."""
    df = pd.read_csv(path)
    df["review_date"] = pd.to_datetime(df["review_date"])
    return df


# ── Inventory Formulas ────────────────────────────────────────────────────────

def compute_days_to_stockout(stock_quantity: float, avg_daily_sales: float) -> float:
    """
    days_to_stockout = stock_quantity / avg_daily_sales
    Returns 999.0 if avg_daily_sales is 0 (no sales → no imminent stockout).
    """
    if avg_daily_sales == 0:
        return 999.0
    return round(stock_quantity / avg_daily_sales, 2)


def classify_inventory_status(days: float) -> str:
    """
    Critical: < 7 days  |  Low: 7–14 days  |  Healthy: > 14 days
    """
    if days < CRITICAL_DAYS:
        return "Critical"
    elif days <= LOW_DAYS:
        return "Low"
    else:
        return "Healthy"


def compute_revenue_at_risk(
    price: float,
    stock_quantity: float,
    avg_daily_sales: float,
    threshold_days: int = REVENUE_THRESHOLD_DAYS,
) -> float:
    """
    Revenue at risk = price × (remaining stock + avg_daily_sales × threshold_days)
    Represents potential revenue lost if the product stocksout.
    """
    return round(price * (stock_quantity + avg_daily_sales * threshold_days), 2)


# ── Pricing Formulas ──────────────────────────────────────────────────────────

def compute_gross_margin(price: float, cost: float) -> float:
    """
    gross_margin_pct = (price - cost) / price * 100
    Returns 0.0 if price is 0 (guard against division by zero).
    """
    if price == 0:
        return 0.0
    return round((price - cost) / price * 100, 2)


def get_category_avg_prices(df_products: pd.DataFrame) -> dict:
    """Return a dict of {category: avg_price} across all products."""
    return df_products.groupby("category")["price"].mean().to_dict()


def classify_price_positioning(price: float, category_avg: float) -> str:
    """
    Premium  : price > category_avg × 1.20
    Budget   : price < category_avg × 0.80
    Mid-Range: otherwise
    """
    if price > category_avg * PREMIUM_BAND:
        return "Premium"
    elif price < category_avg * BUDGET_BAND:
        return "Budget"
    else:
        return "Mid-Range"


# ── Core Business Logic Functions ─────────────────────────────────────────────

def get_inventory_health_data(product_id: str, df: pd.DataFrame) -> dict:
    """Return full inventory health dict for a given product_id."""
    row = df[df["product_id"] == product_id]
    if row.empty:
        return {"error": f"Product {product_id} not found"}
    r = row.iloc[0]
    days = compute_days_to_stockout(r["stock_quantity"], r["avg_daily_sales"])
    status = classify_inventory_status(days)
    return {
        "product_id": product_id,
        "product_name": r["product_name"],
        "category": r["category"],
        "stock_quantity": int(r["stock_quantity"]),
        "avg_daily_sales": float(r["avg_daily_sales"]),
        "days_to_stockout": days,
        "status": status,
        "reorder_level": int(r["reorder_level"]),
        "below_reorder": bool(r["stock_quantity"] < r["reorder_level"]),
    }


def get_pricing_analysis_data(product_id: str, df: pd.DataFrame) -> dict:
    """Return pricing intelligence dict for a given product_id."""
    row = df[df["product_id"] == product_id]
    if row.empty:
        return {"error": f"Product {product_id} not found"}
    r = row.iloc[0]
    category_avgs = get_category_avg_prices(df)
    category_avg = category_avgs.get(r["category"], r["price"])
    margin = compute_gross_margin(r["price"], r["cost"])
    positioning = classify_price_positioning(r["price"], category_avg)
    return {
        "product_id": product_id,
        "product_name": r["product_name"],
        "category": r["category"],
        "price": float(r["price"]),
        "cost": float(r["cost"]),
        "gross_margin_pct": margin,
        "category_avg_price": round(category_avg, 2),
        "price_positioning": positioning,
        "below_20_margin_flag": bool(margin < LOW_MARGIN_FLAG),
        "below_25_margin_flag": bool(margin < BRIEFING_MARGIN_FLAG),
    }


def get_category_performance_data(category: str, df: pd.DataFrame) -> dict:
    """Return aggregated performance metrics for a product category."""
    cat_df = df[df["category"] == category]
    if cat_df.empty:
        return {"error": f"Category '{category}' not found. Valid: Tops, Dresses, Bottoms, Outerwear, Accessories"}

    # Compute per-product derived fields
    cat_df = cat_df.copy()
    cat_df["gross_margin_pct"] = cat_df.apply(
        lambda r: compute_gross_margin(r["price"], r["cost"]), axis=1
    )
    cat_df["days_to_stockout"] = cat_df.apply(
        lambda r: compute_days_to_stockout(r["stock_quantity"], r["avg_daily_sales"]), axis=1
    )
    cat_df["status"] = cat_df["days_to_stockout"].apply(classify_inventory_status)
    cat_df["daily_revenue"] = cat_df["price"] * cat_df["avg_daily_sales"]

    top3 = (
        cat_df.nlargest(3, "daily_revenue")[
            ["product_id", "product_name", "price", "avg_daily_sales", "daily_revenue"]
        ]
        .round(2)
        .to_dict("records")
    )

    return {
        "category": category,
        "total_skus": int(len(cat_df)),
        "avg_rating": round(cat_df["avg_rating"].mean(), 2),
        "avg_margin_pct": round(cat_df["gross_margin_pct"].mean(), 2),
        "total_stock_units": int(cat_df["stock_quantity"].sum()),
        "critical_stock_count": int((cat_df["status"] == "Critical").sum()),
        "low_stock_count": int((cat_df["status"] == "Low").sum()),
        "healthy_stock_count": int((cat_df["status"] == "Healthy").sum()),
        "avg_return_rate": round(cat_df["return_rate"].mean(), 3),
        "top_3_revenue_products": top3,
    }


def get_restock_alerts(df: pd.DataFrame, threshold_days: int = 7) -> list:
    """
    Scan all products. Return those with days_to_stockout <= threshold_days,
    sorted ascending by urgency (fewest days first).
    Each entry includes revenue_at_risk.
    """
    results = []
    for _, r in df.iterrows():
        days = compute_days_to_stockout(r["stock_quantity"], r["avg_daily_sales"])
        if days <= threshold_days:
            rev_risk = compute_revenue_at_risk(
                r["price"], r["stock_quantity"], r["avg_daily_sales"], threshold_days
            )
            results.append(
                {
                    "product_id": r["product_id"],
                    "product_name": r["product_name"],
                    "category": r["category"],
                    "stock_quantity": int(r["stock_quantity"]),
                    "avg_daily_sales": float(r["avg_daily_sales"]),
                    "days_to_stockout": days,
                    "status": classify_inventory_status(days),
                    "reorder_level": int(r["reorder_level"]),
                    "revenue_at_risk_inr": rev_risk,
                }
            )
    results.sort(key=lambda x: x["days_to_stockout"])
    return results


def search_products_data(
    query: str, df: pd.DataFrame, category: Optional[str] = None
) -> list:
    """
    Search products by text query across product_name and category.
    Optionally filter by category. Returns top 5 matches as list of dicts.
    Uses case-insensitive substring matching + rapidfuzz scoring.
    """
    try:
        from rapidfuzz import fuzz
        use_fuzzy = True
    except ImportError:
        use_fuzzy = False

    search_df = df.copy()
    if category and category.lower() != "all categories":
        search_df = search_df[search_df["category"].str.lower() == category.lower()]

    query_lower = query.lower()
    scores = []
    for _, r in search_df.iterrows():
        text = f"{r['product_name']} {r['category']}".lower()
        if use_fuzzy:
            score = max(
                fuzz.token_set_ratio(query_lower, text),
                fuzz.WRatio(query_lower, text),
            )
        else:
            score = 100 if query_lower in text else 50 if any(w in text for w in query_lower.split()) else 0
        scores.append((score, r))

    scores.sort(key=lambda x: x[0], reverse=True)
    top5 = scores[:5]

    return [
        {
            "product_id": r["product_id"],
            "product_name": r["product_name"],
            "category": r["category"],
            "price": float(r["price"]),
            "stock_quantity": int(r["stock_quantity"]),
            "avg_rating": float(r["avg_rating"]),
            "match_score": score,
        }
        for score, r in top5
    ]


# ── Daily Briefing Helpers ────────────────────────────────────────────────────

def get_top3_low_stock(df: pd.DataFrame) -> list:
    """Return top 3 most critically low-stock products with revenue at risk."""
    all_alerts = get_restock_alerts(df, threshold_days=999)
    # Sort by days_to_stockout ascending, take top 3
    sorted_all = sorted(all_alerts, key=lambda x: x["days_to_stockout"])
    return sorted_all[:3]


def get_worst_rated_product(df: pd.DataFrame) -> dict:
    """Return the product with the lowest avg_rating."""
    row = df.loc[df["avg_rating"].idxmin()]
    return {
        "product_id": row["product_id"],
        "product_name": row["product_name"],
        "category": row["category"],
        "avg_rating": float(row["avg_rating"]),
        "review_count": int(row["review_count"]),
        "return_rate": float(row["return_rate"]),
    }


def get_lowest_margin_product(df: pd.DataFrame) -> dict:
    """Return the product with the lowest gross margin percentage."""
    df_copy = df.copy()
    df_copy["gross_margin_pct"] = df_copy.apply(
        lambda r: compute_gross_margin(r["price"], r["cost"]), axis=1
    )
    row = df_copy.loc[df_copy["gross_margin_pct"].idxmin()]
    margin = row["gross_margin_pct"]
    return {
        "product_id": row["product_id"],
        "product_name": row["product_name"],
        "category": row["category"],
        "price": float(row["price"]),
        "cost": float(row["cost"]),
        "gross_margin_pct": round(float(margin), 2),
        "below_25_flag": bool(margin < BRIEFING_MARGIN_FLAG),
        "below_20_flag": bool(margin < LOW_MARGIN_FLAG),
    }


# ── Full Analytics Pipeline ───────────────────────────────────────────────────

def run_full_pipeline(df_products: pd.DataFrame, df_reviews: pd.DataFrame) -> dict:
    """
    Run the complete analytics pipeline and return all KPIs for the dashboard.
    Returns a dict with summary metrics, alerts, and per-category data.
    """
    df = df_products.copy()
    df["gross_margin_pct"] = df.apply(
        lambda r: compute_gross_margin(r["price"], r["cost"]), axis=1
    )
    df["days_to_stockout"] = df.apply(
        lambda r: compute_days_to_stockout(r["stock_quantity"], r["avg_daily_sales"]), axis=1
    )
    df["status"] = df["days_to_stockout"].apply(classify_inventory_status)
    df["revenue_at_risk"] = df.apply(
        lambda r: compute_revenue_at_risk(
            r["price"], r["stock_quantity"], r["avg_daily_sales"]
        ),
        axis=1,
    )
    df["daily_revenue"] = df["price"] * df["avg_daily_sales"]

    # Price positioning per product
    cat_avgs = get_category_avg_prices(df)
    df["price_positioning"] = df.apply(
        lambda r: classify_price_positioning(r["price"], cat_avgs.get(r["category"], r["price"])),
        axis=1,
    )

    # Summary KPIs
    critical_df = df[df["status"] == "Critical"]
    low_df = df[df["status"] == "Low"]
    below_20 = df[df["gross_margin_pct"] < LOW_MARGIN_FLAG]

    # Top 10 priority products (most urgent: fewest days to stockout)
    top10 = (
        df.nsmallest(10, "days_to_stockout")[
            [
                "product_id", "product_name", "category", "price",
                "stock_quantity", "avg_daily_sales", "days_to_stockout",
                "status", "gross_margin_pct", "avg_rating", "revenue_at_risk",
            ]
        ]
        .round(2)
        .to_dict("records")
    )

    return {
        "total_skus": int(len(df)),
        "critical_stock_count": int(len(critical_df)),
        "low_stock_count": int(len(low_df)),
        "healthy_stock_count": int((df["status"] == "Healthy").sum()),
        "avg_margin_pct": round(df["gross_margin_pct"].mean(), 2),
        "avg_rating": round(df["avg_rating"].mean(), 2),
        "below_20_margin_count": int(len(below_20)),
        "total_revenue_at_risk_inr": round(critical_df["revenue_at_risk"].sum(), 2),
        "top10_priority": top10,
        "df_enriched": df,   # full enriched DataFrame for charts
    }
