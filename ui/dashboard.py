"""
ui/dashboard.py — Tab 1: Dashboard with KPI cards, alert banners,
top-10 priority table, and Run Full Analysis button.
"""

import streamlit as st
import pandas as pd


def _status_icon(status: str) -> str:
    return {"Critical": "🔴", "Low": "🟡", "Healthy": "🟢"}.get(status, "⚪")


def render_dashboard(pipeline: dict, df_enriched: pd.DataFrame, category_filter: str = "All Categories"):
    """Render the full dashboard tab."""
    st.subheader("📊 StyleCraft Catalog Dashboard")

    # ── Category Filter ────────────────────────────────────────────────────────
    df = df_enriched.copy()
    is_filtered = bool(category_filter and category_filter != "All Categories")
    if is_filtered:
        df = df[df["category"] == category_filter]
        st.info(f"Showing **{category_filter}** category — {len(df)} SKU(s)", icon="📂")

    # ── Derive KPIs from (filtered) df ────────────────────────────────────────
    # Use df_enriched-derived columns (all present after run_full_pipeline):
    #   status, gross_margin_pct, avg_rating, revenue_at_risk
    critical_df = df[df["status"] == "Critical"]
    low_df      = df[df["status"] == "Low"]
    below_20    = df[df["gross_margin_pct"] < 20.0]

    total_skus      = len(df)
    critical_count  = len(critical_df)
    low_count       = len(low_df)
    avg_margin      = round(float(df["gross_margin_pct"].mean()), 1) if total_skus else 0.0
    avg_rating      = round(float(df["avg_rating"].mean()), 2)       if total_skus else 0.0
    rev_at_risk     = round(float(critical_df["revenue_at_risk"].sum()), 0)

    # ── KPI Cards ─────────────────────────────────────────────────────────────
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric(label="📦 Total SKUs", value=total_skus)
    with col2:
        st.metric(
            label="🔴 Critical Stock",
            value=critical_count,
            delta=f"+{low_count} Low",
            delta_color="inverse",
        )
    with col3:
        st.metric(label="💰 Avg Margin %", value=f"{avg_margin:.1f}%")
    with col4:
        st.metric(label="⭐ Avg Rating", value=f"{avg_rating:.2f} / 5.0")

    st.divider()

    # ── Alert Banners ──────────────────────────────────────────────────────────
    if not critical_df.empty:
        names = ", ".join(critical_df["product_name"].tolist())
        st.error(
            f"🔴 **CRITICAL STOCK ALERT** — {critical_count} product(s) will stockout in <7 days: "
            f"**{names}**  |  Total revenue at risk: ₹{rev_at_risk:,.0f}",
            icon="🚨",
        )

    if not low_df.empty:
        names = ", ".join(low_df["product_name"].tolist())
        st.warning(
            f"🟡 **LOW STOCK WARNING** — {low_count} product(s) have 7–14 days of inventory: "
            f"**{names}**",
            icon="⚠️",
        )

    if critical_count == 0 and low_count == 0:
        st.success("🟢 All products have healthy inventory levels (>14 days)", icon="✅")

    if not below_20.empty:
        names = ", ".join(below_20["product_name"].tolist())
        st.info(
            f"💰 **MARGIN ALERT** — {len(below_20)} product(s) have gross margin below 20%: **{names}**",
            icon="📉",
        )

    st.divider()

    # ── Top-10 Priority Table ─────────────────────────────────────────────────
    label = (
        f"🎯 Top {min(10, total_skus)} Priority Products"
        + (f" — {category_filter}" if is_filtered else " (by urgency)")
    )
    st.markdown(f"#### {label}")
    st.caption("Sorted by days to stockout — most urgent first")

    top10 = (
        df.nsmallest(10, "days_to_stockout")[[
            "product_id", "product_name", "category", "price",
            "stock_quantity", "avg_daily_sales", "days_to_stockout",
            "status", "gross_margin_pct", "avg_rating",
        ]].round(2).to_dict("records")
    )

    if top10:
        display_df = pd.DataFrame(top10)
        display_df["Status"] = display_df["status"].apply(
            lambda s: f"{_status_icon(s)} {s}"
        )

        styled = display_df[[
            "product_id", "product_name", "category", "price",
            "stock_quantity", "avg_daily_sales", "days_to_stockout",
            "Status", "gross_margin_pct", "avg_rating",
        ]].rename(columns={
            "product_id": "ID",
            "product_name": "Product Name",
            "category": "Category",
            "price": "Price (₹)",
            "stock_quantity": "Stock",
            "avg_daily_sales": "Daily Sales",
            "days_to_stockout": "Days Left",
            "gross_margin_pct": "Margin %",
            "avg_rating": "Rating",
        })

        def _color_row_dashboard(row):
            # rgba semi-transparent tints — work on dark AND light mode backgrounds.
            # Healthy rows return "" so the theme background is never overridden.
            s = str(row.get("Status", ""))
            if "Critical" in s:
                return ["background-color: rgba(239, 68, 68, 0.20)"] * len(row)
            elif "Low" in s:
                return ["background-color: rgba(245, 158, 11, 0.18)"] * len(row)
            return [""] * len(row)

        st.dataframe(
            styled.style.apply(_color_row_dashboard, axis=1)
                        .format({"Price (₹)": "₹{:,.0f}", "Margin %": "{:.1f}%", "Rating": "{:.1f}"}),
            use_container_width=True,
            height=380,
        )

    st.divider()

    # ── Run Full Analysis Button ────────────────────────────────────────────────
    st.markdown("#### 🤖 Run Full AI Analysis")
    st.caption("Triggers the RetailMind Orchestrator — InventoryAnalyst → BusinessIntelligence")

    if st.button("▶ Run Full Analysis", type="primary", use_container_width=True):
        orchestrator = st.session_state.get("orchestrator")
        if orchestrator is None:
            st.warning("Enter your OpenAI API key in the sidebar to enable AI analysis.", icon="🔑")
            return

        with st.status("🔄 Running RetailMind Orchestrator...", expanded=True) as status_ui:
            try:
                st.write("**Phase 1: InventoryAnalystAgent** — scanning stockout risks...")
                results = orchestrator.run_full_analysis(verbose=False)

                st.write("**Phase 2: BusinessIntelligenceAgent** — analysing margins & reviews...")

                if results.get("errors"):
                    for err in results["errors"]:
                        st.error(f"Error: {err}")

                status_ui.update(label="✅ Analysis Complete!", state="complete")

                # Show results
                st.divider()
                if results.get("inventory_analyst"):
                    with st.expander("📦 Inventory Analysis Results", expanded=True):
                        st.markdown(results["inventory_analyst"])
                if results.get("business_intelligence"):
                    with st.expander("💼 Business Intelligence Results", expanded=True):
                        st.markdown(results["business_intelligence"])

            except Exception as e:
                status_ui.update(label="❌ Analysis failed", state="error")
                st.error(f"Orchestrator error: {e}")
