"""
ui/charts.py — Tab 3: 5 interactive Plotly charts in sub-tabs.

Charts (Plotly only — never matplotlib/seaborn):
  1. Grouped bar — Current price vs category average (positioning)
  2. Horizontal bar — Gross margin % with 20% and 25% threshold lines
  3. Bar — Inventory status (days to stockout, colour-coded by status)
  4. Bar — Sales velocity (avg_daily_sales) with category average line
  5. Scatter — Priority matrix (days to stockout vs margin, bubble = revenue at risk)
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


STATUS_COLORS = {"Critical": "#ef4444", "Low": "#f59e0b", "Healthy": "#22c55e"}


def render_charts(df_enriched: pd.DataFrame, category_filter: str = "All Categories"):
    """Render all 5 Plotly charts in sub-tabs."""
    st.subheader("📈 Visual Analytics")

    # Apply category filter
    df = df_enriched.copy()
    if category_filter and category_filter != "All Categories":
        df = df[df["category"] == category_filter]

    if df.empty:
        st.warning(f"No products found for category: {category_filter}")
        return

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "💲 Price Comparison",
        "📊 Margin Health",
        "📦 Inventory Risk",
        "⚡ Sales Velocity",
        "🎯 Priority Matrix",
    ])

    with tab1:
        _chart_price_comparison(df)

    with tab2:
        _chart_margin_health(df)

    with tab3:
        _chart_inventory_risk(df)

    with tab4:
        _chart_sales_velocity(df)

    with tab5:
        _chart_priority_matrix(df)


def _chart_price_comparison(df: pd.DataFrame):
    """Grouped bar: product price vs category average price."""
    st.markdown("#### Product Price vs Category Average")
    st.caption("Visualise which products are Premium, Mid-Range, or Budget vs their category")

    # Category averages
    cat_avg = df.groupby("category")["price"].mean().reset_index()
    cat_avg.columns = ["category", "category_avg_price"]
    df_merged = df.merge(cat_avg, on="category")

    # Sort by category then price
    df_plot = df_merged.sort_values(["category", "price"])

    fig = go.Figure()
    fig.add_trace(go.Bar(
        name="Product Price (₹)",
        x=df_plot["product_name"],
        y=df_plot["price"],
        marker_color=df_plot["category"].map({
            "Tops": "#6366f1", "Dresses": "#ec4899",
            "Bottoms": "#14b8a6", "Outerwear": "#f97316", "Accessories": "#8b5cf6"
        }).fillna("#64748b"),
        customdata=df_plot[["product_id", "category", "price_positioning"]],
        hovertemplate=(
            "<b>%{x}</b><br>"
            "Price: ₹%{y:,.0f}<br>"
            "Category: %{customdata[1]}<br>"
            "Positioning: %{customdata[2]}"
            "<extra></extra>"
        ),
    ))
    fig.add_trace(go.Bar(
        name="Category Avg (₹)",
        x=df_plot["product_name"],
        y=df_plot["category_avg_price"],
        marker_color="rgba(100,116,139,0.35)",
        hovertemplate="Category Avg: ₹%{y:,.0f}<extra></extra>",
    ))

    fig.update_layout(
        barmode="group",
        xaxis_tickangle=-45,
        xaxis_title="Product",
        yaxis_title="Price (₹)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        height=460,
        margin=dict(b=160),
    )
    st.plotly_chart(fig, use_container_width=True)


def _chart_margin_health(df: pd.DataFrame):
    """Horizontal bar: gross margin % with threshold lines."""
    st.markdown("#### Gross Margin Health")
    st.caption("Target: ≥25% | Alert: <20%")

    df_sorted = df.sort_values("gross_margin_pct", ascending=True)
    colors = df_sorted["gross_margin_pct"].apply(
        lambda m: "#ef4444" if m < 20 else "#f59e0b" if m < 25 else "#22c55e"
    )

    fig = go.Figure(go.Bar(
        x=df_sorted["gross_margin_pct"],
        y=df_sorted["product_name"],
        orientation="h",
        marker_color=colors,
        customdata=df_sorted[["product_id", "price", "cost"]],
        hovertemplate=(
            "<b>%{y}</b><br>"
            "Margin: %{x:.1f}%<br>"
            "Price: ₹%{customdata[1]:,.0f} | Cost: ₹%{customdata[2]:,.0f}"
            "<extra></extra>"
        ),
    ))

    # Threshold reference lines
    fig.add_vline(x=20, line_dash="dash", line_color="#ef4444", line_width=2,
                  annotation_text="20% Alert", annotation_position="top right")
    fig.add_vline(x=25, line_dash="dash", line_color="#f59e0b", line_width=2,
                  annotation_text="25% Target", annotation_position="top right")

    fig.update_layout(
        xaxis_title="Gross Margin %",
        yaxis_title="",
        height=max(400, len(df) * 22),
        margin=dict(l=10, r=40),
    )
    st.plotly_chart(fig, use_container_width=True)


def _chart_inventory_risk(df: pd.DataFrame):
    """Bar chart: days to stockout, colour-coded by Critical/Low/Healthy."""
    st.markdown("#### Inventory Risk — Days to Stockout")
    st.caption("🔴 Critical (<7 days) | 🟡 Low (7-14 days) | 🟢 Healthy (>14 days)")

    # Cap display at 30 for readability
    df_plot = df.sort_values("days_to_stockout", ascending=True).copy()
    df_plot["days_display"] = df_plot["days_to_stockout"].clip(upper=30)
    df_plot["status_label"] = df_plot["days_to_stockout"].apply(
        lambda d: "🔴 Critical" if d < 7 else "🟡 Low" if d <= 14 else "🟢 Healthy"
    )

    fig = px.bar(
        df_plot,
        x="product_name",
        y="days_display",
        color="status",
        color_discrete_map=STATUS_COLORS,
        custom_data=["product_id", "stock_quantity", "avg_daily_sales", "days_to_stockout"],
        labels={"days_display": "Days to Stockout (capped at 30)", "product_name": "Product"},
        height=440,
    )
    fig.update_traces(
        hovertemplate=(
            "<b>%{x}</b><br>"
            "Actual days: %{customdata[3]:.1f}<br>"
            "Stock: %{customdata[1]} units | Daily sales: %{customdata[2]}"
            "<extra></extra>"
        )
    )

    # Critical threshold line
    fig.add_hline(y=7, line_dash="dash", line_color="#ef4444",
                  annotation_text="7-day Critical threshold")
    fig.add_hline(y=14, line_dash="dot", line_color="#f59e0b",
                  annotation_text="14-day Low threshold")

    fig.update_layout(xaxis_tickangle=-45, margin=dict(b=160))
    st.plotly_chart(fig, use_container_width=True)


def _chart_sales_velocity(df: pd.DataFrame):
    """Bar: avg_daily_sales per product with category average line."""
    st.markdown("#### Sales Velocity — Average Daily Units Sold")
    st.caption("Higher velocity + low stock = urgent restock needed")

    df_plot = df.sort_values("avg_daily_sales", ascending=False).copy()
    cat_avg_sales = df_plot["avg_daily_sales"].mean()

    fig = px.bar(
        df_plot,
        x="product_name",
        y="avg_daily_sales",
        color="category",
        custom_data=["product_id", "stock_quantity", "days_to_stockout", "status"],
        labels={"avg_daily_sales": "Avg Daily Sales (units)", "product_name": "Product"},
        color_discrete_sequence=px.colors.qualitative.Set2,
        height=420,
    )
    fig.update_traces(
        hovertemplate=(
            "<b>%{x}</b><br>"
            "Daily sales: %{y:.1f} units<br>"
            "Stock: %{customdata[1]} units | Days left: %{customdata[2]:.1f}"
            "<extra></extra>"
        )
    )

    # Catalog average line
    fig.add_hline(
        y=cat_avg_sales,
        line_dash="dash",
        line_color="#6366f1",
        annotation_text=f"Catalog avg: {cat_avg_sales:.1f}",
        annotation_position="top right",
    )
    fig.update_layout(xaxis_tickangle=-45, margin=dict(b=160))
    st.plotly_chart(fig, use_container_width=True)


def _chart_priority_matrix(df: pd.DataFrame):
    """Scatter: days to stockout vs gross margin, bubble = revenue at risk."""
    st.markdown("#### Priority Matrix — Stockout Risk vs Margin Health")
    st.caption(
        "Bubble size = Revenue at risk (₹) | "
        "Bottom-left = most urgent (low days + low margin)"
    )

    df_plot = df.copy()
    df_plot["days_capped"] = df_plot["days_to_stockout"].clip(upper=30)
    df_plot["rev_display"] = df_plot["revenue_at_risk"].clip(lower=1000)

    fig = px.scatter(
        df_plot,
        x="days_capped",
        y="gross_margin_pct",
        size="rev_display",
        color="status",
        color_discrete_map=STATUS_COLORS,
        hover_name="product_name",
        custom_data=["product_id", "price", "stock_quantity", "revenue_at_risk"],
        labels={
            "days_capped": "Days to Stockout (capped at 30)",
            "gross_margin_pct": "Gross Margin %",
        },
        size_max=55,
        height=480,
    )
    fig.update_traces(
        hovertemplate=(
            "<b>%{hoverName}</b> (%{customdata[0]})<br>"
            "Days to stockout: %{x:.1f}<br>"
            "Gross margin: %{y:.1f}%<br>"
            "Price: ₹%{customdata[1]:,.0f} | Stock: %{customdata[2]}<br>"
            "Revenue at risk: ₹%{customdata[3]:,.0f}"
            "<extra></extra>"
        )
    )

    # Quadrant lines
    fig.add_vline(x=7, line_dash="dash", line_color="#ef4444", line_width=1)
    fig.add_hline(y=25, line_dash="dash", line_color="#f59e0b", line_width=1)

    # Quadrant annotations
    fig.add_annotation(x=2, y=15, text="⚠️ CRITICAL ZONE", showarrow=False,
                        font=dict(color="#ef4444", size=11))
    fig.add_annotation(x=22, y=70, text="✅ SAFE ZONE", showarrow=False,
                        font=dict(color="#22c55e", size=11))

    st.plotly_chart(fig, use_container_width=True)
