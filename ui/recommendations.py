"""
ui/recommendations.py — Tab 4: Inventory recommendations table with
filters, colour-coded rows, rationale expanders, and CSV download.
"""

import streamlit as st
import pandas as pd
from io import StringIO


def _color_row(row):
    """Apply rgba semi-transparent tints by inventory status.
    Uses substring 'in' check to handle emoji-prefixed values like '🔴 Critical'.
    Returns "" for Healthy/unknown rows so the theme background is never overridden.
    Works correctly in both dark mode and light mode."""
    s = str(row.get("Status", ""))
    if "Critical" in s:
        return ["background-color: rgba(239, 68, 68, 0.20)"] * len(row)
    elif "Low" in s:
        return ["background-color: rgba(245, 158, 11, 0.18)"] * len(row)
    return [""] * len(row)


def render_recommendations(df_enriched: pd.DataFrame, category_filter: str = "All Categories"):
    """Render the full recommendations tab."""
    st.subheader("🔔 Inventory Recommendations")
    st.caption("Products ranked by urgency — most critical first")

    df = df_enriched.copy()

    # ── Filters ────────────────────────────────────────────────────────────────
    col1, col2, col3 = st.columns(3)

    with col1:
        status_options = ["All", "Critical", "Low", "Healthy"]
        status_filter = st.multiselect(
            "Filter by Status",
            options=status_options[1:],
            default=[],
            placeholder="All statuses",
            key="rec_status_filter",
        )

    with col2:
        cat_options = ["All Categories", "Tops", "Dresses", "Bottoms", "Outerwear", "Accessories"]

        # Unconditionally pre-set the multiselect's session-state key BEFORE the widget
        # renders. Streamlit uses st.session_state["rec_cat_filter"] as the widget's current
        # value when the key is present — so this makes the multiselect always mirror the
        # sidebar on every rerun with zero stale-state risk.
        st.session_state["rec_cat_filter"] = (
            [category_filter] if category_filter != "All Categories" else []
        )

        rec_cat_filter = st.multiselect(
            "Filter by Category",
            options=cat_options[1:],
            placeholder="All categories",
            key="rec_cat_filter",
        )

    with col3:
        margin_filter = st.checkbox("⚠️ Below 20% Margin only", value=False, key="rec_margin_filter")

    # Apply filters
    if status_filter:
        df = df[df["status"].isin(status_filter)]
    if rec_cat_filter:
        df = df[df["category"].isin(rec_cat_filter)]
    if margin_filter:
        df = df[df["gross_margin_pct"] < 20.0]

    # Sort by urgency
    df = df.sort_values("days_to_stockout", ascending=True)

    if df.empty:
        st.info("No products match the current filters.", icon="🔍")
        return

    st.markdown(f"**Showing {len(df)} product(s)**")
    st.divider()

    # ── Styled Table ───────────────────────────────────────────────────────────
    display_cols = {
        "product_id": "ID",
        "product_name": "Product Name",
        "category": "Category",
        "price": "Price (₹)",
        "stock_quantity": "Stock",
        "avg_daily_sales": "Daily Sales",
        "days_to_stockout": "Days Left",
        "status": "Status",
        "gross_margin_pct": "Margin %",
        "avg_rating": "Rating",
        "revenue_at_risk": "Rev. at Risk (₹)",
    }

    table_df = df[list(display_cols.keys())].rename(columns=display_cols).copy()
    table_df["Status"] = table_df["Status"].apply(
        lambda s: f"{'🔴' if s=='Critical' else '🟡' if s=='Low' else '🟢'} {s}"
    )

    fmt = {
        "Price (₹)": "₹{:,.0f}",
        "Margin %": "{:.1f}%",
        "Rating": "{:.1f}",
        "Rev. at Risk (₹)": "₹{:,.0f}",
        "Days Left": "{:.1f}",
    }

    st.dataframe(
        table_df.style.apply(_color_row, axis=1).format(fmt),
        use_container_width=True,
        height=420,
    )

    st.divider()

    # ── Rationale Expanders ────────────────────────────────────────────────────
    st.markdown("#### 📋 Action Rationale")
    st.caption("Expand each product to see the recommended action and reasoning")

    for _, row in df.head(10).iterrows():
        days = row["days_to_stockout"]
        status = row["status"]
        margin = row["gross_margin_pct"]
        stock = int(row["stock_quantity"])
        reorder = int(row.get("reorder_level", 0))
        priority = 1 if status == "Critical" else 2 if status == "Low" else 3

        # Build action recommendation
        actions = []
        if status == "Critical":
            actions.append(f"🔴 **REORDER IMMEDIATELY** — only {days:.1f} days of stock remain")
        elif status == "Low":
            actions.append(f"🟡 **Place reorder soon** — {days:.1f} days remaining (reorder level: {reorder})")
        if margin < 20:
            actions.append(f"📉 **Review pricing** — margin {margin:.1f}% is below 20% minimum")
        if row.get("return_rate", 0) > 0.20:
            actions.append(f"⚠️ **Investigate returns** — return rate {row['return_rate']*100:.0f}% is high")

        if not actions:
            actions.append(f"✅ No immediate action required — {status.lower()} inventory, {margin:.1f}% margin")

        label = (
            f"[P{priority}] {row['product_id']} — "
            f"{row['product_name']}: {actions[0].replace('**','').split('—')[0].strip()}"
        )

        with st.expander(label):
            st.markdown(f"**Category:** {row['category']}")
            st.markdown(f"**Stock:** {stock} units | **Daily sales:** {row['avg_daily_sales']:.1f}/day")
            st.markdown(f"**Days to stockout:** {days:.1f} | **Revenue at risk:** ₹{row['revenue_at_risk']:,.0f}")
            st.markdown(f"**Gross margin:** {margin:.1f}% | **Avg rating:** {row['avg_rating']:.1f}/5")
            st.markdown("**Recommended actions:**")
            for action in actions:
                st.markdown(f"  • {action}")

    st.divider()

    # ── CSV Download ────────────────────────────────────────────────────────────
    csv_buf = StringIO()
    table_df.to_csv(csv_buf, index=False)
    st.download_button(
        label="⬇️ Download Recommendations CSV",
        data=csv_buf.getvalue(),
        file_name="retailmind_recommendations.csv",
        mime="text/csv",
        use_container_width=True,
        type="primary",
    )
