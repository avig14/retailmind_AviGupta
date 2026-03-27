"""
core/tools.py — LangChain @tool definitions for RetailMind AI Agent.

NON-NEGOTIABLE RULES:
  1. Every @tool MUST return json.dumps({...}, default=str) — never plain text.
  2. Tools are grouped by agent: Group A (InventoryAnalystAgent), Group B (BusinessIntelligenceAgent).
  3. Tool docstrings are the LLM's ONLY guide — precise, specific, informative.

All tools read data from st.session_state (populated during app startup).
"""

import json
import os

from langchain_core.tools import tool

from core import analytics


def _get_dfs():
    """Retrieve DataFrames from Streamlit session state (call-time access)."""
    try:
        import streamlit as st
        return st.session_state["df_products"], st.session_state["df_reviews"]
    except Exception:
        # Fallback for notebook/testing context — load from disk
        df_products = analytics.load_products()
        df_reviews = analytics.load_reviews()
        return df_products, df_reviews


def _get_review_llm():
    """Get a lightweight LLM instance for review summarisation."""
    from langchain_openai import ChatOpenAI
    api_key = os.getenv("OPENAI_API_KEY", "")
    return ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.3,   # slight creativity for natural summaries
        max_tokens=256,    # short summaries only
        api_key=api_key if api_key else None,
    )


def _summarize_reviews(product_id: str, df_products, df_reviews) -> dict:
    """
    Use LLM to generate a 2-sentence sentiment summary and identify
    top 2 positive + top 2 negative themes from review texts.
    Falls back to statistical summary if no reviews exist in sample data.
    """
    # Get aggregate rating from products CSV (authoritative)
    prod_row = df_products[df_products["product_id"] == product_id]
    if prod_row.empty:
        return {"error": f"Product {product_id} not found"}

    p = prod_row.iloc[0]
    avg_rating_agg = float(p["avg_rating"])
    total_reviews_agg = int(p["review_count"])

    # Get sample reviews from reviews CSV
    reviews_df = df_reviews[df_reviews["product_id"] == product_id]

    if reviews_df.empty:
        return {
            "product_id": product_id,
            "product_name": p["product_name"],
            "avg_rating": avg_rating_agg,
            "total_reviews": total_reviews_agg,
            "sentiment_summary": (
                f"Based on {total_reviews_agg} reviews, this product has an average "
                f"rating of {avg_rating_agg}/5. No detailed review text is available "
                f"in the current sample dataset for deeper analysis."
            ),
            "positive_themes": ["Overall satisfaction with product quality"],
            "negative_themes": ["Insufficient review sample for theme extraction"],
        }

    # Build review text for LLM
    review_texts = []
    for _, rev in reviews_df.iterrows():
        review_texts.append(
            f"[{rev['rating']}★] {rev['review_title']}: {rev['review_text']}"
        )
    combined_text = "\n".join(review_texts)

    prompt = f"""Analyse these customer reviews for '{p['product_name']}' and respond in JSON:

Reviews:
{combined_text}

Respond ONLY with valid JSON in this exact format:
{{
  "sentiment_summary": "<2 sentences summarising overall customer sentiment>",
  "positive_themes": ["<theme 1>", "<theme 2>"],
  "negative_themes": ["<theme 1>", "<theme 2>"]
}}"""

    try:
        llm = _get_review_llm()
        from langchain_core.messages import HumanMessage
        response = llm.invoke([HumanMessage(content=prompt)])
        import re
        json_match = re.search(r"\{.*\}", response.content, re.DOTALL)
        parsed = json.loads(json_match.group()) if json_match else {}
    except Exception as e:
        parsed = {}

    return {
        "product_id": product_id,
        "product_name": p["product_name"],
        "category": p["category"],
        "avg_rating": avg_rating_agg,
        "total_reviews": total_reviews_agg,
        "sample_reviews_analysed": int(len(reviews_df)),
        "sentiment_summary": parsed.get(
            "sentiment_summary",
            f"Rating: {avg_rating_agg}/5 from {total_reviews_agg} reviews."
        ),
        "positive_themes": parsed.get("positive_themes", []),
        "negative_themes": parsed.get("negative_themes", []),
    }


# ── Group A: InventoryAnalystAgent — 3 tools ──────────────────────────────────

@tool
def search_products(query: str, category: str = None) -> str:
    """Search the StyleCraft product catalog (30 SKUs) by text query with optional
    category filter. Returns top 5 matching products with product_id, product_name,
    category, price (INR), stock_quantity, and avg_rating (1-5).
    Valid categories: Tops, Dresses, Bottoms, Outerwear, Accessories.
    Use when the user asks to find a product, browse the catalog, or discover items
    by name, type, or description."""
    df_products, _ = _get_dfs()
    results = analytics.search_products_data(query, df_products, category)
    return json.dumps(
        {"matches": results, "count": len(results), "query": query, "category_filter": category},
        default=str,
    )


@tool
def get_inventory_health(product_id: str) -> str:
    """Get real-time inventory health for a specific StyleCraft product.
    Input: product_id in format 'SC001' through 'SC030'.
    Returns: current stock_quantity, avg_daily_sales, days_to_stockout,
    inventory status (Critical <7 days / Low 7-14 days / Healthy >14 days),
    reorder_level, and whether stock is below reorder threshold.
    Formula: days_to_stockout = stock_quantity / avg_daily_sales.
    Use when asked about stock levels, how long inventory will last,
    stockout risk, or reorder needs for a specific product."""
    df_products, _ = _get_dfs()
    result = analytics.get_inventory_health_data(product_id, df_products)
    return json.dumps(result, default=str)


@tool
def generate_restock_alert(threshold_days: int = 7) -> str:
    """Scan all 30 StyleCraft products and return those at risk of stockout
    within threshold_days (default 7). Results sorted by urgency — fewest
    days remaining first. Each item includes: product_id, product_name, category,
    stock_quantity, avg_daily_sales, days_to_stockout, status, and
    revenue_at_risk_inr (formula: price × (stock + avg_daily_sales × threshold_days)).
    Use when asked about urgent restocking needs, inventory alerts, which products
    need immediate attention, or an overview of at-risk stock."""
    df_products, _ = _get_dfs()
    alerts = analytics.get_restock_alerts(df_products, threshold_days)
    total_rev_risk = sum(a["revenue_at_risk_inr"] for a in alerts)
    return json.dumps(
        {
            "alerts": alerts,
            "threshold_days": threshold_days,
            "alert_count": len(alerts),
            "total_revenue_at_risk_inr": round(total_rev_risk, 2),
        },
        default=str,
    )


# ── Group B: BusinessIntelligenceAgent — 3 tools ─────────────────────────────

@tool
def get_pricing_analysis(product_id: str) -> str:
    """Get pricing intelligence and margin analysis for a specific StyleCraft product.
    Input: product_id in format 'SC001' through 'SC030'.
    Returns: price (INR), cost (INR), gross_margin_pct
    (formula: (price-cost)/price×100), category_avg_price, price_positioning
    (Premium if >120% of category avg / Budget if <80% / Mid-Range otherwise),
    below_20_margin_flag (boolean — action required if True),
    below_25_margin_flag (boolean — monitoring recommended if True).
    Use when asked about margins, profitability, pricing tiers,
    cost efficiency, or whether a product is over/under-priced."""
    df_products, _ = _get_dfs()
    result = analytics.get_pricing_analysis_data(product_id, df_products)
    return json.dumps(result, default=str)


@tool
def get_review_insights(product_id: str) -> str:
    """Get LLM-powered customer review analysis for a StyleCraft product.
    Input: product_id in format 'SC001' through 'SC030'.
    Returns: avg_rating (1.0-5.0), total_reviews (platform total), a 2-sentence
    sentiment_summary generated by AI, top 2 positive_themes (what customers love),
    and top 2 negative_themes (what customers complain about).
    The summary is generated from actual review text in the sample dataset.
    Use when asked about customer feedback, star ratings, complaints, product quality
    perception, what customers are saying, or review sentiment."""
    df_products, df_reviews = _get_dfs()
    result = _summarize_reviews(product_id, df_products, df_reviews)
    return json.dumps(result, default=str)


@tool
def get_category_performance(category: str) -> str:
    """Get aggregated performance metrics for a StyleCraft product category.
    Input: one of 'Tops', 'Dresses', 'Bottoms', 'Outerwear', 'Accessories'.
    Returns: total_skus, avg_rating, avg_margin_pct, total_stock_units,
    critical_stock_count, low_stock_count, healthy_stock_count, avg_return_rate,
    and top_3_revenue_products (by price × avg_daily_sales).
    Use when asked about category overview, which category performs best,
    top revenue generators, or broad catalog discovery by category."""
    df_products, _ = _get_dfs()
    result = analytics.get_category_performance_data(category, df_products)
    return json.dumps(result, default=str)


# ── Tool Collections ──────────────────────────────────────────────────────────

# All tools — used by ChatRouterAgent (Streamlit conversational agent)
ALL_TOOLS = [
    search_products,
    get_inventory_health,
    generate_restock_alert,
    get_pricing_analysis,
    get_review_insights,
    get_category_performance,
]

# Group A — InventoryAnalystAgent (notebook)
INVENTORY_TOOLS = [search_products, get_inventory_health, generate_restock_alert]

# Group B — BusinessIntelligenceAgent (notebook)
BI_TOOLS = [get_pricing_analysis, get_review_insights, get_category_performance]
